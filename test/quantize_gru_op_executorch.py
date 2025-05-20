import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type, List

from torch import fx
from torch import nn
from torch._ops import OpOverload
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer import DerivedQuantizationSpec, Quantizer, SharedQuantizationSpec
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
)

from executorch.backends.cadence.aot.quantizer.utils import (
    find_sequential_partitions_aten,
    is_annotated,
    no_outside_users,
)
from executorch.backends.cadence.aot.quantizer.utils import get_bias_qparams


@dataclass
class NXPPartitionAnchors:
    """
    All fields are lists of (node, input_node), where node is from
    the given partition and input_node is an input to the partition.

    Quantizer uses inputs, weights and biases for quantization annotation. The others
    field contains tensor inputs that aren't quantized, and the literals fields contains
    is used for other types of input values as well as handling default parameters.
    """

    inputs: list[tuple[fx.Node, fx.Node]] = field(default_factory=list)
    weights: list[tuple[fx.Node, fx.Node]] = field(default_factory=list)
    biases: list[tuple[fx.Node, fx.Node] | tuple[fx.Node, fx.Node, DerivedQuantizationSpec]] = field(default_factory=list)
    others: list[tuple[fx.Node, fx.Node]] = field(default_factory=list)
    literals: list[tuple[fx.Node, fx.Node]] = field(default_factory=list)
    outputs: list[tuple[fx.Node, fx.Node] | tuple[fx.Node, fx.Node, SharedQuantizationSpec]] = field(default_factory=list)


class NXPQuantizationPattern(ABC):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        """
        List of types to be passed to find_sequential_partitions_aten.
        """
        pass

    @abstractmethod
    def get_anchors(
            self, gm: torch.fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> Optional[NXPPartitionAnchors]:
        pass


class GruInputPattern(NXPQuantizationPattern):
    """
    Quantization pattern for Gru Input quantization. Accepts 2 input nodes.

    Basic quantization for all inputs and outputs.
    """

    def partition_types(self) -> List[Type[OpOverload]]:
        return [torch.ops.aten.gru.input]

    def get_anchors(
            self, gm: fx.GraphModule, fused_partition: List[fx.GraphModule]
    ) -> NXPPartitionAnchors | None:
        gru_node = fused_partition[0].nodes[-1]
        hidden_state = gru_node.args[1]
        inputs = [(gru_node, gru_node.args[0]), (gru_node, hidden_state)]
        weights = [(gru_node, node) for node in gru_node.args[2] if "weight" in node.target]
        weights_edges = [(x, y) for y, x in weights]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (gru_node.args[0], gru_node),
                (hidden_state, gru_node),
                *weights_edges,
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=torch.per_tensor_affine,
        )
        biases = [(gru_node, node, bias_qspec) for node in gru_node.args[2] if "bias" in node.target]
        outputs = [(gru_node, user) for user in gru_node.users]

        return NXPPartitionAnchors(
            inputs=inputs,
            weights=weights,
            biases=biases,
            outputs=outputs,
        )


class NXPAtenQuantizer(Quantizer):
    def __init__(
            self, pattern: NXPQuantizationPattern, quantization_config: QuantizationConfig
    ) -> None:
        super().__init__()
        self.pattern = pattern
        self.quantization_config = quantization_config

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fused_partitions = find_sequential_partitions_aten(
            model,
            self.pattern.partition_types(),
        )

        input_act_qspec = self.quantization_config.input_activation
        weight_qspec = self.quantization_config.weight
        bias_qspec = self.quantization_config.bias
        output_act_qspec = self.quantization_config.output_activation

        for fused_partition in fused_partitions:
            if not no_outside_users(fused_partition):
                continue

            anchors = self.pattern.get_anchors(model, fused_partition)
            if not anchors:
                continue
            if is_annotated(
                    [
                        x[0]
                        for x in anchors.inputs
                                 + anchors.weights
                                 + anchors.biases
                                 + anchors.outputs
                    ]
            ):
                continue

            for node, output_node, *custom_spec in anchors.outputs:
                # pyre-ignore[16]: no attribute
                annotation = node.meta.get(
                    "quantization_annotation",
                    QuantizationAnnotation(_annotated=True),
                )
                # pyre-ignore[16]: no attribute
                annotation.output_qspec_map[output_node] = (
                    custom_spec[0] if custom_spec else output_act_qspec
                )
                # pyre-ignore[16]: no attribute
                node.meta["quantization_annotation"] = annotation

            def annotate_inputs(
                    inputs: list[tuple[fx.Node , fx.Node]] | list[tuple[fx.Node, fx.Node, DerivedQuantizationSpec]],
                    spec: Optional[QuantizationSpec],
            ) -> None:
                for node, input_node, *custom_spec in inputs:
                    # pyre-ignore[16]: no attribute
                    annotation = node.meta.get(
                        "quantization_annotation",
                        QuantizationAnnotation(_annotated=True),
                    )
                    # pyre-ignore[16]: no attribute
                    annotation.input_qspec_map[input_node] = (
                        custom_spec[0] if custom_spec else spec
                    )
                    # pyre-ignore[16]: no attribute
                    node.meta["quantization_annotation"] = annotation

            annotate_inputs(anchors.inputs, input_act_qspec)
            annotate_inputs(anchors.weights, weight_qspec)
            # pyre-ignore[6]: incompatible parameter type
            annotate_inputs(anchors.biases, bias_qspec)
        return model

    def validate(self, model: fx.GraphModule) -> None:
        pass


act_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2 ** -12),
)

wgt_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
    ch_axis=0
)

class CustomComposableQuantizer(ComposableQuantizer):
    def __init__(self):
        static_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_qspec,
            None,
        )
        super().__init__([NXPAtenQuantizer(GruInputPattern(), static_qconfig), ])

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for quantizer in self.quantizers:
            quantizer.annotate(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        return super().validate(model)


def test_gru_model():
    input_shape = (1, 64)
    model = nn.GRU(64, hidden_size=16, bidirectional=False, bias=False, num_layers=1)
    random_tensors  = (torch.randn(input_shape),)
    calibration_inputs = [random_tensors, random_tensors]
    example_input = (torch.ones(input_shape),)

    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

    quantizer = CustomComposableQuantizer()

    m = prepare_pt2e(exir_program_aten, quantizer)
    for i, data in enumerate(calibration_inputs):
        m(*data)
    m = convert_pt2e(m)

    print(m) # quantized gru - QDQ cluster
