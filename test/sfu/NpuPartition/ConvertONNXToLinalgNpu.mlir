#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "act_test_quant_sym"} {
  func.func @main_graph(%arg0: tensor<1x32xf32> {onnx.name = "input"}) -> (tensor<1x32xf32> {onnx.name = "output"}) {
    %0 = onnx.Constant dense<[-9.82335652E-4, -0.0923864468, 0.0308713373, 0.125592053, -0.0481653772, 0.0492569357, -0.0528857373, 0.191135094, -0.102464393, 0.0097812349, -0.0817779526, 0.00841632765, -0.0643123686, -0.125691578, -0.048106797, -0.115775935, -0.125293836, 7.638620e-02, 0.0334553234, -0.0247902982, 0.0771105587, 0.103332363, -0.129471868, 0.00529868389, 0.0113818403, -0.0332146063, 0.0631280169, 0.0246195775, -0.123717859, -0.0606194735, 0.0602609515, -0.0785476937]> : tensor<32xf32>
    %1 = onnx.Constant dense<[-0.174824119, 0.0794920623, -0.0248840358, -0.109353125, -0.0772407278, -0.0361809246, 0.219115898, 0.187830746, 0.245069295, -0.0233103242, 0.0372669809, -0.0510918386, 0.044748459, -0.00205091923, -0.0826271325, -0.157617912, 0.100945644, 0.106719315, 0.0168068167, -0.0849939808, -0.0280704256, -0.0105170989, 0.18064329, -0.0139956269, -0.035048712, 0.0254881084, -0.0581126623, -0.12114203, 0.0639609322, -0.0395348035, -0.140444294, -0.0503093414]> : tensor<32xf32>
    %2 = onnx.Constant dense<0> : tensor<i8>
    %3 = onnx.Constant dense<0.0227836408> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = onnx.Constant dense<0.00787401571> : tensor<f32>
    %6 = onnx.Constant dense<127> : tensor<32xi8>
    %7 = onnx.Constant dense<0> : tensor<i8>
    %8 = onnx.Constant dense<0.0231352877> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = onnx.Constant dense<0.0230970979> : tensor<f32>
    %11 = onnx.Constant dense<0> : tensor<i8>
    %12 = onnx.Constant dense<0.00787401571> : tensor<f32>
    %13 = onnx.Constant dense<0> : tensor<32xi32>
    %14 = onnx.Constant dense<1.79398747E-4> : tensor<1xf32>
    %15 = onnx.Constant dense<0> : tensor<i32>
    %16 = "onnx.Add"(%arg0, %0) {onnx_node_name = "node_add"} : (tensor<1x32xf32>, tensor<32xf32>) -> tensor<1x32xf32>
    %17 = "onnx.DequantizeLinear"(%13, %14, %15) {axis = 1 : si64, onnx_node_name = "ln.bias_DequantizeLinear"} : (tensor<32xi32>, tensor<1xf32>, tensor<i32>) -> tensor<32xf32>
    %18 = "onnx.DequantizeLinear"(%6, %5, %4) {axis = 1 : si64, onnx_node_name = "ln.weight_DequantizeLinear"} : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>
    %19 = "onnx.QuantizeLinear"(%16, %3, %2) {axis = 1 : si64, onnx_node_name = "add_QuantizeLinear", saturate = 1 : si64} : (tensor<1x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x32xi8>
    %20 = scf.execute_region -> tensor<1x32xi8> {
      %25 = bufferization.alloc_tensor() : tensor<1x32xi8>
      %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "npu_layernorm"} ins(%19 : tensor<1x32xi8>) outs(%25 : tensor<1x32xi8>) attrs =  {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, in_scale = 0.0227836408 : f32, in_zp = 0 : i32, npu.target = "npu", out_scale = 0.0231352877 : f32, out_zp = 0 : i32} {
      ^bb0(%in: i8, %out: i8):
        %27 = arith.addi %in, %in : i8
        linalg.yield %27 : i8
      } -> tensor<1x32xi8>
      scf.yield %26 : tensor<1x32xi8>
    }
    %21 = scf.execute_region -> tensor<1x32xi8> {
      %25 = bufferization.alloc_tensor() : tensor<1x32xi8>
      %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "npu_gelu"} ins(%20 : tensor<1x32xi8>) outs(%25 : tensor<1x32xi8>) attrs =  {in_scale = 0.0231352877 : f32, in_zp = 0 : i32, npu.target = "npu", out_scale = 0.0230970979 : f32, out_zp = 0 : i16} {
      ^bb0(%in: i8, %out: i8):
        %27 = arith.addi %in, %in : i8
        linalg.yield %27 : i8
      } -> tensor<1x32xi8>
      scf.yield %26 : tensor<1x32xi8>
    }
    %22 = scf.execute_region -> tensor<1x32xi8> {
      %25 = bufferization.alloc_tensor() : tensor<1x32xi8>
      %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "npu_softmax"} ins(%21 : tensor<1x32xi8>) outs(%25 : tensor<1x32xi8>) attrs =  {axis = -1 : i64, in_scale = 0.0230970979 : f32, in_zp = 0 : i32, npu.target = "npu", out_scale = 0.00787401571 : f32, out_zp = 0 : i16} {
      ^bb0(%in: i8, %out: i8):
        %27 = arith.addi %in, %in : i8
        linalg.yield %27 : i8
      } -> tensor<1x32xi8>
      scf.yield %26 : tensor<1x32xi8>
    }
    %23 = "onnx.DequantizeLinear"(%22, %12, %11) {axis = 1 : si64, onnx_node_name = "softmax_DequantizeLinear"} : (tensor<1x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x32xf32>
    %24 = "onnx.Add"(%23, %1) {onnx_node_name = "node_add_1"} : (tensor<1x32xf32>, tensor<32xf32>) -> tensor<1x32xf32>
    return %24 : tensor<1x32xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

