#map = affine_map<(d0) -> (-d0 + 1, 0)>
#map1 = affine_map<(d0) -> (-d0 + 225, 3)>
#map2 = affine_map<()[s0, s1] -> (s0 * 224 + s1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "oneshotbufferize"} {
  func.func private @npu_mem_alloc(i64) -> !llvm.ptr
  func.func private @npu_dma_mvin(!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)
  func.func private @npu_sfu_run(i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16)
  func.func private @npu_dma_mvout(!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)
  func.func private @npu_init() -> i32
  func.func @main_graph(%arg0: memref<1x3x224x224xf32> {onnx.name = "input"}) -> (memref<1x16x224x224xf32> {onnx.name = "output"}) attributes {llvm.emit_c_interface} {
    %c3211264_i64 = arith.constant 3211264 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = call @npu_init() : () -> i32
    %1 = "krnl.global"() {name = "constant_0", shape = [16, 3, 3, 3], value = dense<"0xA140863D6E88E9BC9E5930BCF51FFF3DEAF14C3DADC72CBEBD0ABBBC955E23BC3A935ABD291B36BD51EA10BC19FCECBD93182A3D91CB01BE3B0C003E4FBF96BDED9D94BDEC6A3CBD82023EBD62F10F3E0C7612BEADC988BDE13FF8BDE8479C3D50FA0DBE647E3D3E1C58353D90B51DBE1FB131BEC370A2BD8A4CCE3B3A48A6BDF0E1963D16A6B83DC3D944BE781A97BAF32EAEBDB833C8BD155233BD42A716BE5A1192BD9E0ECE3C83BAAC3DB257333E08771F3EB776C4BD14EA0C3E46D8393E74671F3EBF94503DD29F2CBE4124FE3C097189BDF0D9AF3DE801283E51C5FDBD3ED8413D6A57EB3CAEC9233EA9768A3C237C3BBEDA8B313D4922F23D30EE1EBE8B500FBE4D88403E26ED34BE6EBE2A3E8240CA3D8AD8C43DCF3A42BE400C14BE01FBEC3BEC88EBBA0393853DB33C873DAD2F87BCDCB72B3EB06E25BDDE35D23DB85332BE0626983D88BAEA3D3515423EE50990BD7538CB3DA63A36BDF071FDBDBEA3E73C3C188D3DBB46E2BDBA82373E68F0183E759BAC3CEC651BBEFD86A2BC11DB23BEE14034BE07F9293E0E80193EBEEE3A3E6CFDD1BD7316383E2373F93DA5871C3DF19881BD458320BE95784A3D2631A1BB030DFDBB12F5B6BD0FF2BCBD5A4412BED9F8093CC0AD033D095D093D8DD2D8BDBA2C0CBD045FB9BC32A1B03DFE08EBBCC656FC3DF85F78BAED899BBC639096BD98ECAF3B12E943BC06F738BD7D98B2BD125D20BD52694CBDF59F303D9984C83C1932DA3D51D2033E094BA53D152EF5BD0E1353BD542DFDBD86AD29BD11C29C3D035B5B3D527D94BD9D7BCA3CD241B1BDF31B523DE975383DC6480C3DDC14423E078014BE5D9E53BDC9EC4F3D999F743D420F0FBED6209B3DA91315BE4B97EC3D4E481BBE94CB18BE135E2CBE4E4F8A3C3D1A9E3D6D26A23D5516173CD08CF33DC8A9123E5262DC3DA90ABE3D08CF013E8F1FA93D8FEB25BE257C1A3E1CAF33BE99F9BABD7C8CFFBDAC1AC7BDF84E1C3E8DC916BEBAE7AEBDCFA1413E646B69BDFA15EABC83E32DBD3487973DC63218BEF4FB143E36F3C3BDFA8A3B3ECAA0023E8F7B50BDB766B93C17ED1CBCC93A43BE4DF6B3BDCF06A8BD47FF2FBED9FCE5BDDD5FA8BC29AEFB3D3C10073EA884FFBD60D4083E57430C3E336E233E6A9A943DC38D923C7488C0BD45DBB23D0B74EBBD8CE4FA3DDE2F8E3C613D0DBE915EA8BD0D75393CC420423EF5263CBE37C908BEE1D2103EFA9BDD3DBD9CF8BC479E383E51A1B9BD6209FA3DCA6004BE6FC606BEF11820BE12E9EEBDD8D89F3DA414F33D210A09BEB28BCCBC69B63E3E6586F9BDBE9A583D1C1DECBB50AD94BD5545BABC16DC433ED2EFA33D622B26BDACC78CBD24E2583DB42FB63BD937383E64D448BDBFF5ADBD8749CABCE37B2B3CBC2FD4BD91E7533D5A6C9CBD687D3F3EF67324BE4C5D97BC9A8695BD342922BEE487EFBCB24DC93DE52A1ABE6BFF2A3E342F37BE296017BD94683FBEA5495B3D88BE703D91836C3DE9092DBE3CEC333CB29B3C3E4B113DBECB7E383EB4DD09BE0F9124BD02B6893D1981313EE780923D955AB4BD10A3033E3EDBE0BDA04612BDA10628BEF1F93A3E3A6D343DEC6048BD09C6253D2EF5443DA5260B3EBB412A3E9180113E87F5253E40BB443D5FBD7A3DE34D303E8355AEBD82F8D03D6D7EC8BD0C33B8BD4FA96C3D783F3B3E8F550B3E887F29BEF419393DFA66353C986CA53CA55569BD220CEDBBC33E3DBE4313E23D8FC22EBE85F9023E1AA93C3E4F7D063ED3613DBE5C7FF0BD49F472BD4BA4183D8BE3A73D2D343CBE84E9DA3C9CAA343E71F3253EE2828CBDAC9F14BDA46F0FBE0D073D3E100AEDBCC50B423D5B7D8BBC1B15303E682C14BEE4E82BBE40AD3C3E369AAB3D2E09933D5D98E03D2B6B02BE1003A7BDEFD077BD4B7595BD8AB6C1BC90053EBD2E7C81BBF003A5BDBA0ADDBDE67722BE5DF5023ECB4639BE0B87C7BC33D639BEB31E8ABDBC6E333E05502D3ED487EB3D40701ABED769D5BD9AD93FBD395B96BD871A353DFA265B3DF5A2A6BDD58517BE9F0B453D242ECC3D10E813BEF5ED9CBC1A9142BE10D5EB3D77EDBE3C185853BDE49F05BEF0D4F93BC373BC3D2D0B383D8039A53DA969C9BDE66907BE6059FDBD84121CBE3B6879BDE9F0AD3C9825C73D652D383E583AC03D145FC23C39591B3E16BFC4BDEAFF26BE15063A3E6EC9E0BDD383B43D88FD9A3D702EC4BD88C562BC805BFE3D6C5D6ABDDD5D103E5003163E1F86A2BD3A4F36BE18DAA43C5C7F50BDBB1DA9BDA888B5BBD32B313EE77266BD609B90BD800F7B3CA022303D1B48A9BDF5F975BC712673BCACFD34BEF9A5283ED77DAA3D2C9021BE014C123EC73F1A3C87AE04BD23C62DBD929428BEC79DF9BD5BD1C73D9333923D68B5613C07DB41BE3B47203E177E0ABE4E2B0C3E26B443BCA1F1973B7830A0BD07371FBE"> : tensor<16x3x3x3xf32>} : () -> memref<16x3x3x3xf32>
    %2 = "krnl.global"() {name = "constant_1", shape = [16], value = dense<[-0.00388818281, 0.145530269, 0.0235318113, 0.147666618, -0.0342001803, 0.0597546808, 0.0461084396, 6.931200e-02, 0.110225968, -0.189460516, 0.186317548, -0.0136532616, -0.0724048316, 0.054180067, -0.073133856, -0.0355957299]> : tensor<16xf32>} : () -> memref<16xf32>
    %3 = call @npu_mem_alloc(%c3211264_i64) : (i64) -> !llvm.ptr
    %4 = builtin.unrealized_conversion_cast %3 : !llvm.ptr to memref<1x16x224x224xf32, 2>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          affine.for %arg4 = 0 to 224 {
            affine.for %arg5 = 0 to 224 {
              %7 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %cst) -> (f32) {
                %10 = affine.for %arg8 = max #map(%arg4) to min #map1(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %11 = affine.for %arg10 = max #map(%arg5) to min #map1(%arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %12 = affine.load %arg0[%arg1, %arg6 + %arg2 * 3, %arg8 + %arg4 - 1, %arg10 + %arg5 - 1] : memref<1x3x224x224xf32>
                    %13 = affine.load %1[%arg2 * 16 + %arg3, %arg6, %arg8, %arg10] : memref<16x3x3x3xf32>
                    %14 = arith.mulf %12, %13 : f32
                    %15 = arith.addf %arg11, %14 : f32
                    affine.yield %15 : f32
                  }
                  affine.yield %11 : f32
                }
                affine.yield %10 : f32
              }
              %8 = affine.load %2[%arg2 * 16 + %arg3] : memref<16xf32>
              %9 = arith.addf %7, %8 : f32
              affine.store %9, %4[%arg1, %arg2 * 16 + %arg3, %arg4, %arg5] : memref<1x16x224x224xf32, 2>
            }
          }
        }
      }
    }
    %c2_i8 = arith.constant 2 : i8
    %c65536_i32 = arith.constant 65536 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1_i16 = arith.constant 1 : i16
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c224_i16 = arith.constant 224 : i16
    %c32_i16 = arith.constant 32 : i16
    %c31_i16 = arith.constant 31 : i16
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c224 = arith.constant 224 : index
    %c32 = arith.constant 32 : index
    %c3211264_i64_0 = arith.constant 3211264 : i64
    %5 = call @npu_mem_alloc(%c3211264_i64_0) : (i64) -> !llvm.ptr
    %6 = builtin.unrealized_conversion_cast %5 : !llvm.ptr to memref<1x16x224x224xf32, 2>
    scf.for %arg1 = %c0 to %c224 step %c32 {
      scf.for %arg2 = %c0 to %c224 step %c32 {
        %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %4 : memref<1x16x224x224xf32, 2> -> memref<f32, 2>, index, index, index, index, index, index, index, index, index
        %7 = affine.apply #map2()[%arg1, %arg2]
        %8 = builtin.unrealized_conversion_cast %base_buffer : memref<f32, 2> to !llvm.ptr
        %9 = arith.muli %7, %c4 : index
        %10 = arith.index_cast %9 : index to i64
        %11 = llvm.getelementptr %8[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        func.call @npu_dma_mvin(%11, %c0_i32, %c31_i16, %c31_i16, %c32_i16, %c224_i16, %c1_i8, %c0_i8, %c0_i8, %false, %c0_i32, %c1_i16, %c0_i16) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
        func.call @npu_sfu_run(%c1_i8, %c2_i8, %false, %c0_i32, %c31_i16, %c31_i16, %c65536_i32, %c0_i32, %c0_i16, %c1_i16, %c0_i16, %c1_i16, %c0_i16) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()
        %base_buffer_1, %offset_2, %sizes_3:4, %strides_4:4 = memref.extract_strided_metadata %6 : memref<1x16x224x224xf32, 2> -> memref<f32, 2>, index, index, index, index, index, index, index, index, index
        %12 = affine.apply #map2()[%arg1, %arg2]
        %13 = builtin.unrealized_conversion_cast %base_buffer_1 : memref<f32, 2> to !llvm.ptr
        %14 = arith.muli %12, %c4 : index
        %15 = arith.index_cast %14 : index to i64
        %16 = llvm.getelementptr %13[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        func.call @npu_dma_mvout(%16, %c65536_i32, %c31_i16, %c31_i16, %c32_i16, %c224_i16, %c1_i8, %c0_i8, %c0_i8, %false, %c0_i32, %c1_i16, %c0_i16) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
      }
    }
    %memspacecast = memref.memory_space_cast %6 : memref<1x16x224x224xf32, 2> to memref<1x16x224x224xf32>
    return %memspacecast : memref<1x16x224x224xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 224 , 224] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 224 , 224] , \22name\22 : \22output\22 }\0A\0A]\00"} : () -> ()
}

