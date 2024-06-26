// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65535})" %s | FileCheck %s

func.func @slice_dot_general_lhs(%1181 : tensor<1x16x1x20x100xbf16>, %482 : tensor<1x1x20x16x123xbf16>) -> tensor<1x25x1x123xbf16> {
  %1205 = stablehlo.dot_general %1181, %482, batching_dims = [0, 2] x [0, 1], contracting_dims = [3, 1] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x16x1x20x100xbf16>, tensor<1x1x20x16x123xbf16>) -> tensor<1x1x100x123xbf16>
  %rs = stablehlo.reshape %1205 : (tensor<1x1x100x123xbf16>) -> tensor<1x100x1x123xbf16>
  %1208 = stablehlo.slice %rs [0:1, 75:100, 0:1, 0:123] : (tensor<1x100x1x123xbf16>) -> tensor<1x25x1x123xbf16>
  return %1208 : tensor<1x25x1x123xbf16>
}

func.func @slice_dot_general_batch(%1181 : tensor<12x16x1x20x100xbf16>, %482 : tensor<12x1x20x16x123xbf16>) -> tensor<4x100x1x123xbf16> {
  %1205 = stablehlo.dot_general %1181, %482, batching_dims = [0, 2] x [0, 1], contracting_dims = [3, 1] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<12x16x1x20x100xbf16>, tensor<12x1x20x16x123xbf16>) -> tensor<12x1x100x123xbf16>
  %rs = stablehlo.reshape %1205 : (tensor<12x1x100x123xbf16>) -> tensor<12x100x1x123xbf16>
  %1208 = stablehlo.slice %rs [2:10:2, 0:100, 0:1, 0:123] : (tensor<12x100x1x123xbf16>) -> tensor<4x100x1x123xbf16>
  return %1208 : tensor<4x100x1x123xbf16>
}

// CHECK:  func.func @slice_dot_general_lhs(%arg0: tensor<1x16x1x20x100xbf16>, %arg1: tensor<1x1x20x16x123xbf16>) -> tensor<1x25x1x123xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:16, 0:1, 0:20, 75:100] : (tensor<1x16x1x20x100xbf16>) -> tensor<1x16x1x20x25xbf16>
// CHECK-NEXT:    %1 = stablehlo.dot_general %0, %arg1, batching_dims = [0, 2] x [0, 1], contracting_dims = [3, 1] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x16x1x20x25xbf16>, tensor<1x1x20x16x123xbf16>) -> tensor<1x1x25x123xbf16>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1x1x25x123xbf16>) -> tensor<1x25x1x123xbf16>
// CHECK-NEXT:    return %2 : tensor<1x25x1x123xbf16>
// CHECK-NEXT:  }
// CHECK:  func.func @slice_dot_general_batch(%arg0: tensor<12x16x1x20x100xbf16>, %arg1: tensor<12x1x20x16x123xbf16>) -> tensor<4x100x1x123xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [2:10:2, 0:16, 0:1, 0:20, 0:100] : (tensor<12x16x1x20x100xbf16>) -> tensor<4x16x1x20x100xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [2:10:2, 0:1, 0:20, 0:16, 0:123] : (tensor<12x1x20x16x123xbf16>) -> tensor<4x1x20x16x123xbf16>
// CHECK-NEXT:    %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 2] x [0, 1], contracting_dims = [3, 1] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<4x16x1x20x100xbf16>, tensor<4x1x20x16x123xbf16>) -> tensor<4x1x100x123xbf16>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<4x1x100x123xbf16>) -> tensor<4x100x1x123xbf16>
// CHECK-NEXT:    return %3 : tensor<4x100x1x123xbf16>
// CHECK-NEXT:  }

