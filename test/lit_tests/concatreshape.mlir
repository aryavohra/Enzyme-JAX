// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<3x4xf32>, %b : tensor<3x4xf32>) -> tensor<2x3x4xf32> {
    %u = stablehlo.reshape %a : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %v = stablehlo.reshape %b : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %concat = stablehlo.concatenate %u, %v, dim=0 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<2x3x4xf32>
    return %concat : tensor<2x3x4xf32>
  }

  func.func @main2(%a : tensor<3x4xf32>, %b : tensor<3x4xf32>) -> tensor<2x3x4xf64> {
    %u = stablehlo.reshape %a : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %uc = stablehlo.convert %u : (tensor<1x3x4xf32>) -> tensor<1x3x4xf64>
    %v = stablehlo.reshape %b : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %vc = stablehlo.convert %v : (tensor<1x3x4xf32>) -> tensor<1x3x4xf64>
    %concat = stablehlo.concatenate %uc, %vc, dim=0 : (tensor<1x3x4xf64>, tensor<1x3x4xf64>) -> tensor<2x3x4xf64>
    return %concat : tensor<2x3x4xf64>
  }

  // TODO this opt
  func.func @main3(%a : tensor<3x4xf32>, %b : tensor<3x4xf32>) -> tensor<3x2x4xf32> {
    %u = stablehlo.reshape %a : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    %v = stablehlo.reshape %b : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    %concat = stablehlo.concatenate %u, %v, dim=1 : (tensor<3x1x4xf32>, tensor<3x1x4xf32>) -> tensor<3x2x4xf32>
    return %concat : tensor<3x2x4xf32>
  }
}


// CHECK:  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<6x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:    return %1 : tensor<2x3x4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x3x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3x4xf32>) -> tensor<3x4xf64>
// CHECK-NEXT:    %1 = stablehlo.convert %arg1 : (tensor<3x4xf32>) -> tensor<3x4xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<6x4xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<6x4xf64>) -> tensor<2x3x4xf64>
// CHECK-NEXT:    return %3 : tensor<2x3x4xf64>
// CHECK-NEXT:  }