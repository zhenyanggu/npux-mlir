module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "oneshotbufferize"} {
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.mlir.global external constant @_entry_point_1_oneshotbufferize("run_main_graph_oneshotbufferize\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_in_sig_oneshotbufferize("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 224 , 224] , \22name\22 : \22input\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_out_sig_oneshotbufferize("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 224 , 224] , \22name\22 : \22output\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_oneshotbufferize("run_main_graph\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_in_sig_oneshotbufferize("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 224 , 224] , \22name\22 : \22input\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_out_sig_oneshotbufferize("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 224 , 224] , \22name\22 : \22output\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.func @omGetExternalConstantAddr(!llvm.ptr, !llvm.ptr, i64)
  llvm.func @omMMapBinaryFile(!llvm.ptr, !llvm.ptr, i64, i64) -> i1
  llvm.func @omTensorListGetSize(!llvm.ptr) -> i64
  llvm.func @omTensorPrint(!llvm.ptr, !llvm.ptr)
  llvm.func @omTensorListGetOmtArray(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorSetDataType(!llvm.ptr, i64)
  llvm.func @omTensorGetDataType(!llvm.ptr) -> i64
  llvm.func @omTensorGetStrides(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetShape(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetRank(!llvm.ptr) -> i64
  llvm.func @omTensorSetDataPtr(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr)
  llvm.func @omTensorGetDataPtr(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorDestroy(!llvm.ptr)
  llvm.func @omTensorCreateUntyped(i64) -> !llvm.ptr
  llvm.func @omTensorListCreate(!llvm.ptr, i64) -> !llvm.ptr
  llvm.mlir.global internal constant @constant_1_oneshotbufferize(dense<[-0.00388818281, 0.145530269, 0.0235318113, 0.147666618, -0.0342001803, 0.0597546808, 0.0461084396, 6.931200e-02, 0.110225968, -0.189460516, 0.186317548, -0.0136532616, -0.0724048316, 0.054180067, -0.073133856, -0.0355957299]> : tensor<16xf32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<16 x f32>
  llvm.mlir.global internal constant @constant_0_oneshotbufferize("\A1@\86=n\88\E9\BC\9EY0\BC\F5\1F\FF=\EA\F1L=\AD\C7,\BE\BD\0A\BB\BC\95^#\BC:\93Z\BD)\1B6\BDQ\EA\10\BC\19\FC\EC\BD\93\18*=\91\CB\01\BE;\0C\00>O\BF\96\BD\ED\9D\94\BD\ECj<\BD\82\02>\BDb\F1\0F>\0Cv\12\BE\AD\C9\88\BD\E1?\F8\BD\E8G\9C=P\FA\0D\BEd~=>\1CX5=\90\B5\1D\BE\1F\B11\BE\C3p\A2\BD\8AL\CE;:H\A6\BD\F0\E1\96=\16\A6\B8=\C3\D9D\BEx\1A\97\BA\F3.\AE\BD\B83\C8\BD\15R3\BDB\A7\16\BEZ\11\92\BD\9E\0E\CE<\83\BA\AC=\B2W3>\08w\1F>\B7v\C4\BD\14\EA\0C>F\D89>tg\1F>\BF\94P=\D2\9F,\BEA$\FE<\09q\89\BD\F0\D9\AF=\E8\01(>Q\C5\FD\BD>\D8A=jW\EB<\AE\C9#>\A9v\8A<#|;\BE\DA\8B1=I\22\F2=0\EE\1E\BE\8BP\0F\BEM\88@>&\ED4\BEn\BE*>\82@\CA=\8A\D8\C4=\CF:B\BE@\0C\14\BE\01\FB\EC;\EC\88\EB\BA\03\93\85=\B3<\87=\AD/\87\BC\DC\B7+>\B0n%\BD\DE5\D2=\B8S2\BE\06&\98=\88\BA\EA=5\15B>\E5\09\90\BDu8\CB=\A6:6\BD\F0q\FD\BD\BE\A3\E7<<\18\8D=\BBF\E2\BD\BA\827>h\F0\18>u\9B\AC<\ECe\1B\BE\FD\86\A2\BC\11\DB#\BE\E1@4\BE\07\F9)>\0E\80\19>\BE\EE:>l\FD\D1\BDs\168>#s\F9=\A5\87\1C=\F1\98\81\BDE\83 \BE\95xJ=&1\A1\BB\03\0D\FD\BB\12\F5\B6\BD\0F\F2\BC\BDZD\12\BE\D9\F8\09<\C0\AD\03=\09]\09=\8D\D2\D8\BD\BA,\0C\BD\04_\B9\BC2\A1\B0=\FE\08\EB\BC\C6V\FC=\F8_x\BA\ED\89\9B\BCc\90\96\BD\98\EC\AF;\12\E9C\BC\06\F78\BD}\98\B2\BD\12] \BDRiL\BD\F5\9F0=\99\84\C8<\192\DA=Q\D2\03>\09K\A5=\15.\F5\BD\0E\13S\BDT-\FD\BD\86\AD)\BD\11\C2\9C=\03[[=R}\94\BD\9D{\CA<\D2A\B1\BD\F3\1BR=\E9u8=\C6H\0C=\DC\14B>\07\80\14\BE]\9ES\BD\C9\ECO=\99\9Ft=B\0F\0F\BE\D6 \9B=\A9\13\15\BEK\97\EC=NH\1B\BE\94\CB\18\BE\13^,\BENO\8A<=\1A\9E=m&\A2=U\16\17<\D0\8C\F3=\C8\A9\12>Rb\DC=\A9\0A\BE=\08\CF\01>\8F\1F\A9=\8F\EB%\BE%|\1A>\1C\AF3\BE\99\F9\BA\BD|\8C\FF\BD\AC\1A\C7\BD\F8N\1C>\8D\C9\16\BE\BA\E7\AE\BD\CF\A1A>dki\BD\FA\15\EA\BC\83\E3-\BD4\87\97=\C62\18\BE\F4\FB\14>6\F3\C3\BD\FA\8A;>\CA\A0\02>\8F{P\BD\B7f\B9<\17\ED\1C\BC\C9:C\BEM\F6\B3\BD\CF\06\A8\BDG\FF/\BE\D9\FC\E5\BD\DD_\A8\BC)\AE\FB=<\10\07>\A8\84\FF\BD`\D4\08>WC\0C>3n#>j\9A\94=\C3\8D\92<t\88\C0\BDE\DB\B2=\0Bt\EB\BD\8C\E4\FA=\DE/\8E<a=\0D\BE\91^\A8\BD\0Du9<\C4 B>\F5&<\BE7\C9\08\BE\E1\D2\10>\FA\9B\DD=\BD\9C\F8\BCG\9E8>Q\A1\B9\BDb\09\FA=\CA`\04\BEo\C6\06\BE\F1\18 \BE\12\E9\EE\BD\D8\D8\9F=\A4\14\F3=!\0A\09\BE\B2\8B\CC\BCi\B6>>e\86\F9\BD\BE\9AX=\1C\1D\EC\BBP\AD\94\BDUE\BA\BC\16\DCC>\D2\EF\A3=b+&\BD\AC\C7\8C\BD$\E2X=\B4/\B6;\D978>d\D4H\BD\BF\F5\AD\BD\87I\CA\BC\E3{+<\BC/\D4\BD\91\E7S=Zl\9C\BDh}?>\F6s$\BEL]\97\BC\9A\86\95\BD4)\22\BE\E4\87\EF\BC\B2M\C9=\E5*\1A\BEk\FF*>4/7\BE)`\17\BD\94h?\BE\A5I[=\88\BEp=\91\83l=\E9\09-\BE<\EC3<\B2\9B<>K\11=\BE\CB~8>\B4\DD\09\BE\0F\91$\BD\02\B6\89=\19\811>\E7\80\92=\95Z\B4\BD\10\A3\03>>\DB\E0\BD\A0F\12\BD\A1\06(\BE\F1\F9:>:m4=\EC`H\BD\09\C6%=.\F5D=\A5&\0B>\BBA*>\91\80\11>\87\F5%>@\BBD=_\BDz=\E3M0>\83U\AE\BD\82\F8\D0=m~\C8\BD\0C3\B8\BDO\A9l=x?;>\8FU\0B>\88\7F)\BE\F4\199=\FAf5<\98l\A5<\A5Ui\BD\22\0C\ED\BB\C3>=\BEC\13\E2=\8F\C2.\BE\85\F9\02>\1A\A9<>O}\06>\D3a=\BE\\\7F\F0\BDI\F4r\BDK\A4\18=\8B\E3\A7=-4<\BE\84\E9\DA<\9C\AA4>q\F3%>\E2\82\8C\BD\AC\9F\14\BD\A4o\0F\BE\0D\07=>\10\0A\ED\BC\C5\0BB=[}\8B\BC\1B\150>h,\14\BE\E4\E8+\BE@\AD<>6\9A\AB=.\09\93=]\98\E0=+k\02\BE\10\03\A7\BD\EF\D0w\BDKu\95\BD\8A\B6\C1\BC\90\05>\BD.|\81\BB\F0\03\A5\BD\BA\0A\DD\BD\E6w\22\BE]\F5\02>\CBF9\BE\0B\87\C7\BC3\D69\BE\B3\1E\8A\BD\BCn3>\05P->\D4\87\EB=@p\1A\BE\D7i\D5\BD\9A\D9?\BD9[\96\BD\87\1A5=\FA&[=\F5\A2\A6\BD\D5\85\17\BE\9F\0BE=$.\CC=\10\E8\13\BE\F5\ED\9C\BC\1A\91B\BE\10\D5\EB=w\ED\BE<\18XS\BD\E4\9F\05\BE\F0\D4\F9;\C3s\BC=-\0B8=\809\A5=\A9i\C9\BD\E6i\07\BE`Y\FD\BD\84\12\1C\BE;hy\BD\E9\F0\AD<\98%\C7=e-8>X:\C0=\14_\C2<9Y\1B>\16\BF\C4\BD\EA\FF&\BE\15\06:>n\C9\E0\BD\D3\83\B4=\88\FD\9A=p.\C4\BD\88\C5b\BC\80[\FE=l]j\BD\DD]\10>P\03\16>\1F\86\A2\BD:O6\BE\18\DA\A4<\\\7FP\BD\BB\1D\A9\BD\A8\88\B5\BB\D3+1>\E7rf\BD`\9B\90\BD\80\0F{<\A0\220=\1BH\A9\BD\F5\F9u\BCq&s\BC\AC\FD4\BE\F9\A5(>\D7}\AA=,\90!\BE\01L\12>\C7?\1A<\87\AE\04\BD#\C6-\BD\92\94(\BE\C7\9D\F9\BD[\D1\C7=\933\92=h\B5a<\07\DBA\BE;G >\17~\0A\BEN+\0C>&\B4C\BC\A1\F1\97;x0\A0\BD\077\1F\BE") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.func private @npu_mem_alloc(%arg0: i64) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_npu_mem_alloc(%arg0) : (i64) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @_mlir_ciface_npu_mem_alloc(i64) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @npu_dma_mvin(%arg0: !llvm.ptr, %arg1: i32, %arg2: i16, %arg3: i16, %arg4: i16, %arg5: i16, %arg6: i8, %arg7: i8, %arg8: i8, %arg9: i1, %arg10: i32, %arg11: i16, %arg12: i16) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    llvm.call @_mlir_ciface_npu_dma_mvin(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_npu_dma_mvin(!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @npu_sfu_run(%arg0: i8, %arg1: i8, %arg2: i1, %arg3: i32, %arg4: i16, %arg5: i16, %arg6: i32, %arg7: i32, %arg8: i16, %arg9: i16, %arg10: i16, %arg11: i16, %arg12: i16) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    llvm.call @_mlir_ciface_npu_sfu_run(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_npu_sfu_run(i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @npu_dma_mvout(%arg0: !llvm.ptr, %arg1: i32, %arg2: i16, %arg3: i16, %arg4: i16, %arg5: i16, %arg6: i8, %arg7: i8, %arg8: i8, %arg9: i1, %arg10: i32, %arg11: i16, %arg12: i16) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    llvm.call @_mlir_ciface_npu_dma_mvout(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_npu_dma_mvout(!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @npu_init() -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_npu_init() : () -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_npu_init() -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @main_graph_oneshotbufferize(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg5, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg9, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg10, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.constant(3211264 : i64) : i64
    %13 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %14 = llvm.call @npu_init() : () -> i32
    %15 = llvm.mlir.addressof @constant_0_oneshotbufferize : !llvm.ptr
    %16 = llvm.bitcast %15 : !llvm.ptr to !llvm.ptr
    %17 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %18 = llvm.insertvalue %16, %17[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %16, %18[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.insertvalue %20, %19[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.mlir.constant(16 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.constant(27 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.mlir.constant(3 : index) : i64
    %27 = llvm.insertvalue %26, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.mlir.constant(9 : index) : i64
    %29 = llvm.insertvalue %28, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.mlir.constant(3 : index) : i64
    %31 = llvm.insertvalue %30, %29[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.mlir.constant(3 : index) : i64
    %33 = llvm.insertvalue %32, %31[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.mlir.constant(3 : index) : i64
    %35 = llvm.insertvalue %34, %33[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.insertvalue %36, %35[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %38 = llvm.mlir.addressof @constant_1_oneshotbufferize : !llvm.ptr
    %39 = llvm.bitcast %38 : !llvm.ptr to !llvm.ptr
    %40 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %39, %40[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.insertvalue %39, %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.mlir.constant(0 : index) : i64
    %44 = llvm.insertvalue %43, %42[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.mlir.constant(16 : index) : i64
    %46 = llvm.insertvalue %45, %44[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.insertvalue %47, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.call @npu_mem_alloc(%12) : (i64) -> !llvm.ptr
    %50 = builtin.unrealized_conversion_cast %49 : !llvm.ptr to memref<1x16x224x224xf32, 2>
    %51 = builtin.unrealized_conversion_cast %50 : memref<1x16x224x224xf32, 2> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%52 : i64)
  ^bb1(%55: i64):  // 2 preds: ^bb0, ^bb23
    %56 = llvm.icmp "slt" %55, %53 : i64
    llvm.cond_br %56, ^bb2, ^bb24
  ^bb2:  // pred: ^bb1
    %57 = llvm.mlir.constant(0 : index) : i64
    %58 = llvm.mlir.constant(1 : index) : i64
    %59 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%57 : i64)
  ^bb3(%60: i64):  // 2 preds: ^bb2, ^bb22
    %61 = llvm.icmp "slt" %60, %58 : i64
    llvm.cond_br %61, ^bb4, ^bb23
  ^bb4:  // pred: ^bb3
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.mlir.constant(16 : index) : i64
    %64 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%62 : i64)
  ^bb5(%65: i64):  // 2 preds: ^bb4, ^bb21
    %66 = llvm.icmp "slt" %65, %63 : i64
    llvm.cond_br %66, ^bb6, ^bb22
  ^bb6:  // pred: ^bb5
    %67 = llvm.mlir.constant(0 : index) : i64
    %68 = llvm.mlir.constant(224 : index) : i64
    %69 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%67 : i64)
  ^bb7(%70: i64):  // 2 preds: ^bb6, ^bb20
    %71 = llvm.icmp "slt" %70, %68 : i64
    llvm.cond_br %71, ^bb8, ^bb21
  ^bb8:  // pred: ^bb7
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.mlir.constant(224 : index) : i64
    %74 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb9(%72 : i64)
  ^bb9(%75: i64):  // 2 preds: ^bb8, ^bb19
    %76 = llvm.icmp "slt" %75, %73 : i64
    llvm.cond_br %76, ^bb10, ^bb20
  ^bb10:  // pred: ^bb9
    %77 = llvm.mlir.constant(0 : index) : i64
    %78 = llvm.mlir.constant(3 : index) : i64
    %79 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb11(%77, %13 : i64, f32)
  ^bb11(%80: i64, %81: f32):  // 2 preds: ^bb10, ^bb18
    %82 = llvm.icmp "slt" %80, %78 : i64
    llvm.cond_br %82, ^bb12, ^bb19
  ^bb12:  // pred: ^bb11
    %83 = llvm.mlir.constant(-1 : index) : i64
    %84 = llvm.mul %70, %83 overflow<nsw> : i64
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.add %84, %85 : i64
    %87 = llvm.mlir.constant(0 : index) : i64
    %88 = llvm.intr.smax(%86, %87) : (i64, i64) -> i64
    %89 = llvm.mlir.constant(-1 : index) : i64
    %90 = llvm.mul %70, %89 overflow<nsw> : i64
    %91 = llvm.mlir.constant(225 : index) : i64
    %92 = llvm.add %90, %91 : i64
    %93 = llvm.mlir.constant(3 : index) : i64
    %94 = llvm.intr.smin(%92, %93) : (i64, i64) -> i64
    %95 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%88, %81 : i64, f32)
  ^bb13(%96: i64, %97: f32):  // 2 preds: ^bb12, ^bb17
    %98 = llvm.icmp "slt" %96, %94 : i64
    llvm.cond_br %98, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %99 = llvm.mlir.constant(-1 : index) : i64
    %100 = llvm.mul %75, %99 overflow<nsw> : i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.add %100, %101 : i64
    %103 = llvm.mlir.constant(0 : index) : i64
    %104 = llvm.intr.smax(%102, %103) : (i64, i64) -> i64
    %105 = llvm.mlir.constant(-1 : index) : i64
    %106 = llvm.mul %75, %105 overflow<nsw> : i64
    %107 = llvm.mlir.constant(225 : index) : i64
    %108 = llvm.add %106, %107 : i64
    %109 = llvm.mlir.constant(3 : index) : i64
    %110 = llvm.intr.smin(%108, %109) : (i64, i64) -> i64
    %111 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%104, %97 : i64, f32)
  ^bb15(%112: i64, %113: f32):  // 2 preds: ^bb14, ^bb16
    %114 = llvm.icmp "slt" %112, %110 : i64
    llvm.cond_br %114, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %115 = llvm.mlir.constant(3 : index) : i64
    %116 = llvm.mul %60, %115 overflow<nsw> : i64
    %117 = llvm.add %80, %116 : i64
    %118 = llvm.add %96, %70 : i64
    %119 = llvm.mlir.constant(-1 : index) : i64
    %120 = llvm.add %118, %119 : i64
    %121 = llvm.add %112, %75 : i64
    %122 = llvm.mlir.constant(-1 : index) : i64
    %123 = llvm.add %121, %122 : i64
    %124 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %125 = llvm.mlir.constant(150528 : index) : i64
    %126 = llvm.mul %55, %125 overflow<nsw, nuw> : i64
    %127 = llvm.mlir.constant(50176 : index) : i64
    %128 = llvm.mul %117, %127 overflow<nsw, nuw> : i64
    %129 = llvm.add %126, %128 overflow<nsw, nuw> : i64
    %130 = llvm.mlir.constant(224 : index) : i64
    %131 = llvm.mul %120, %130 overflow<nsw, nuw> : i64
    %132 = llvm.add %129, %131 overflow<nsw, nuw> : i64
    %133 = llvm.add %132, %123 overflow<nsw, nuw> : i64
    %134 = llvm.getelementptr inbounds|nuw %124[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %135 = llvm.load %134 : !llvm.ptr -> f32
    %136 = llvm.mlir.constant(16 : index) : i64
    %137 = llvm.mul %60, %136 overflow<nsw> : i64
    %138 = llvm.add %137, %65 : i64
    %139 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %140 = llvm.mlir.constant(27 : index) : i64
    %141 = llvm.mul %138, %140 overflow<nsw, nuw> : i64
    %142 = llvm.mlir.constant(9 : index) : i64
    %143 = llvm.mul %80, %142 overflow<nsw, nuw> : i64
    %144 = llvm.add %141, %143 overflow<nsw, nuw> : i64
    %145 = llvm.mlir.constant(3 : index) : i64
    %146 = llvm.mul %96, %145 overflow<nsw, nuw> : i64
    %147 = llvm.add %144, %146 overflow<nsw, nuw> : i64
    %148 = llvm.add %147, %112 overflow<nsw, nuw> : i64
    %149 = llvm.getelementptr inbounds|nuw %139[%148] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %150 = llvm.load %149 : !llvm.ptr -> f32
    %151 = llvm.fmul %135, %150 : f32
    %152 = llvm.fadd %113, %151 : f32
    %153 = llvm.add %112, %111 : i64
    llvm.br ^bb15(%153, %152 : i64, f32)
  ^bb17:  // pred: ^bb15
    %154 = llvm.add %96, %95 : i64
    llvm.br ^bb13(%154, %113 : i64, f32)
  ^bb18:  // pred: ^bb13
    %155 = llvm.add %80, %79 : i64
    llvm.br ^bb11(%155, %97 : i64, f32)
  ^bb19:  // pred: ^bb11
    %156 = llvm.mlir.constant(16 : index) : i64
    %157 = llvm.mul %60, %156 overflow<nsw> : i64
    %158 = llvm.add %157, %65 : i64
    %159 = llvm.extractvalue %48[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %160 = llvm.getelementptr inbounds|nuw %159[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %161 = llvm.load %160 : !llvm.ptr -> f32
    %162 = llvm.fadd %81, %161 : f32
    %163 = llvm.mlir.constant(16 : index) : i64
    %164 = llvm.mul %60, %163 overflow<nsw> : i64
    %165 = llvm.add %164, %65 : i64
    %166 = llvm.extractvalue %51[1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %167 = llvm.mlir.constant(802816 : index) : i64
    %168 = llvm.mul %55, %167 overflow<nsw, nuw> : i64
    %169 = llvm.mlir.constant(50176 : index) : i64
    %170 = llvm.mul %165, %169 overflow<nsw, nuw> : i64
    %171 = llvm.add %168, %170 overflow<nsw, nuw> : i64
    %172 = llvm.mlir.constant(224 : index) : i64
    %173 = llvm.mul %70, %172 overflow<nsw, nuw> : i64
    %174 = llvm.add %171, %173 overflow<nsw, nuw> : i64
    %175 = llvm.add %174, %75 overflow<nsw, nuw> : i64
    %176 = llvm.getelementptr inbounds|nuw %166[%175] : (!llvm.ptr<2>, i64) -> !llvm.ptr<2>, f32
    llvm.store %162, %176 : f32, !llvm.ptr<2>
    %177 = llvm.add %75, %74 : i64
    llvm.br ^bb9(%177 : i64)
  ^bb20:  // pred: ^bb9
    %178 = llvm.add %70, %69 : i64
    llvm.br ^bb7(%178 : i64)
  ^bb21:  // pred: ^bb7
    %179 = llvm.add %65, %64 : i64
    llvm.br ^bb5(%179 : i64)
  ^bb22:  // pred: ^bb5
    %180 = llvm.add %60, %59 : i64
    llvm.br ^bb3(%180 : i64)
  ^bb23:  // pred: ^bb3
    %181 = llvm.add %55, %54 : i64
    llvm.br ^bb1(%181 : i64)
  ^bb24:  // pred: ^bb1
    %182 = llvm.mlir.constant(2 : i8) : i8
    %183 = llvm.mlir.constant(65536 : i32) : i32
    %184 = llvm.mlir.constant(0 : i16) : i16
    %185 = llvm.mlir.constant(1 : i16) : i16
    %186 = llvm.mlir.constant(false) : i1
    %187 = llvm.mlir.constant(0 : i8) : i8
    %188 = llvm.mlir.constant(1 : i8) : i8
    %189 = llvm.mlir.constant(224 : i16) : i16
    %190 = llvm.mlir.constant(32 : i16) : i16
    %191 = llvm.mlir.constant(31 : i16) : i16
    %192 = llvm.mlir.constant(0 : i32) : i32
    %193 = llvm.mlir.constant(4 : index) : i64
    %194 = llvm.mlir.constant(0 : index) : i64
    %195 = llvm.mlir.constant(224 : index) : i64
    %196 = llvm.mlir.constant(32 : index) : i64
    %197 = llvm.mlir.constant(3211264 : i64) : i64
    %198 = llvm.call @npu_mem_alloc(%197) : (i64) -> !llvm.ptr
    %199 = builtin.unrealized_conversion_cast %198 : !llvm.ptr to memref<1x16x224x224xf32, 2>
    %200 = builtin.unrealized_conversion_cast %199 : memref<1x16x224x224xf32, 2> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    llvm.br ^bb25(%194 : i64)
  ^bb25(%201: i64):  // 2 preds: ^bb24, ^bb29
    %202 = llvm.icmp "slt" %201, %195 : i64
    llvm.cond_br %202, ^bb26, ^bb30
  ^bb26:  // pred: ^bb25
    llvm.br ^bb27(%194 : i64)
  ^bb27(%203: i64):  // 2 preds: ^bb26, ^bb28
    %204 = llvm.icmp "slt" %203, %195 : i64
    llvm.cond_br %204, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %205 = llvm.extractvalue %51[0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %206 = llvm.extractvalue %51[1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %207 = llvm.mlir.poison : !llvm.struct<(ptr<2>, ptr<2>, i64)>
    %208 = llvm.insertvalue %205, %207[0] : !llvm.struct<(ptr<2>, ptr<2>, i64)> 
    %209 = llvm.insertvalue %206, %208[1] : !llvm.struct<(ptr<2>, ptr<2>, i64)> 
    %210 = llvm.mlir.constant(0 : index) : i64
    %211 = llvm.insertvalue %210, %209[2] : !llvm.struct<(ptr<2>, ptr<2>, i64)> 
    %212 = builtin.unrealized_conversion_cast %211 : !llvm.struct<(ptr<2>, ptr<2>, i64)> to memref<f32, 2>
    %213 = llvm.extractvalue %51[2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %214 = llvm.extractvalue %51[3, 0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %215 = llvm.extractvalue %51[3, 1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %216 = llvm.extractvalue %51[3, 2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %217 = llvm.extractvalue %51[3, 3] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %218 = llvm.extractvalue %51[4, 0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %219 = llvm.extractvalue %51[4, 1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %220 = llvm.extractvalue %51[4, 2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %221 = llvm.extractvalue %51[4, 3] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %222 = llvm.mlir.constant(224 : index) : i64
    %223 = llvm.mul %201, %222 overflow<nsw> : i64
    %224 = llvm.add %223, %203 : i64
    %225 = builtin.unrealized_conversion_cast %212 : memref<f32, 2> to !llvm.ptr
    %226 = llvm.mul %224, %193 : i64
    %227 = llvm.getelementptr %225[%226] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.call @npu_dma_mvin(%227, %192, %191, %191, %190, %189, %188, %187, %187, %186, %192, %185, %184) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.call @npu_sfu_run(%188, %182, %186, %192, %191, %191, %183, %192, %184, %185, %184, %185, %184) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()
    %228 = llvm.extractvalue %200[0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %229 = llvm.extractvalue %200[1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %230 = llvm.mlir.poison : !llvm.struct<(ptr<2>, ptr<2>, i64)>
    %231 = llvm.insertvalue %228, %230[0] : !llvm.struct<(ptr<2>, ptr<2>, i64)> 
    %232 = llvm.insertvalue %229, %231[1] : !llvm.struct<(ptr<2>, ptr<2>, i64)> 
    %233 = llvm.mlir.constant(0 : index) : i64
    %234 = llvm.insertvalue %233, %232[2] : !llvm.struct<(ptr<2>, ptr<2>, i64)> 
    %235 = builtin.unrealized_conversion_cast %234 : !llvm.struct<(ptr<2>, ptr<2>, i64)> to memref<f32, 2>
    %236 = llvm.extractvalue %200[2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %237 = llvm.extractvalue %200[3, 0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %238 = llvm.extractvalue %200[3, 1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %239 = llvm.extractvalue %200[3, 2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %240 = llvm.extractvalue %200[3, 3] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %241 = llvm.extractvalue %200[4, 0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %242 = llvm.extractvalue %200[4, 1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %243 = llvm.extractvalue %200[4, 2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %244 = llvm.extractvalue %200[4, 3] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %245 = llvm.mlir.constant(224 : index) : i64
    %246 = llvm.mul %201, %245 overflow<nsw> : i64
    %247 = llvm.add %246, %203 : i64
    %248 = builtin.unrealized_conversion_cast %235 : memref<f32, 2> to !llvm.ptr
    %249 = llvm.mul %247, %193 : i64
    %250 = llvm.getelementptr %248[%249] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.call @npu_dma_mvout(%250, %183, %191, %191, %190, %189, %188, %187, %187, %186, %192, %185, %184) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    %251 = llvm.add %203, %196 : i64
    llvm.br ^bb27(%251 : i64)
  ^bb29:  // pred: ^bb27
    %252 = llvm.add %201, %196 : i64
    llvm.br ^bb25(%252 : i64)
  ^bb30:  // pred: ^bb25
    %253 = llvm.extractvalue %200[0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %254 = llvm.extractvalue %200[1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %255 = llvm.extractvalue %200[2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %256 = llvm.extractvalue %200[3, 0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %257 = llvm.extractvalue %200[3, 1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %258 = llvm.extractvalue %200[3, 2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %259 = llvm.extractvalue %200[3, 3] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %260 = llvm.extractvalue %200[4, 0] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %261 = llvm.extractvalue %200[4, 1] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %262 = llvm.extractvalue %200[4, 2] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %263 = llvm.extractvalue %200[4, 3] : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> 
    %264 = llvm.addrspacecast %253 : !llvm.ptr<2> to !llvm.ptr
    %265 = llvm.addrspacecast %254 : !llvm.ptr<2> to !llvm.ptr
    %266 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %267 = llvm.insertvalue %264, %266[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %268 = llvm.insertvalue %265, %267[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %269 = llvm.insertvalue %255, %268[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %270 = llvm.insertvalue %256, %269[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %271 = llvm.insertvalue %260, %270[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %272 = llvm.insertvalue %257, %271[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %273 = llvm.insertvalue %261, %272[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %274 = llvm.insertvalue %258, %273[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %275 = llvm.insertvalue %262, %274[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %276 = llvm.insertvalue %259, %275[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %277 = llvm.insertvalue %263, %276[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.return %277 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph_oneshotbufferize(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.call @main_graph_oneshotbufferize(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %12, %arg0 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @run_main_graph_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr %0[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %4 = llvm.load %3 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %7 = llvm.call @omTensorGetDataPtr(%4) : (!llvm.ptr) -> !llvm.ptr
    %8 = llvm.bitcast %7 : !llvm.ptr to !llvm.ptr
    %9 = llvm.insertvalue %8, %6[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.mlir.constant(0 : i64) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %13 = llvm.call @omTensorGetShape(%4) : (!llvm.ptr) -> !llvm.ptr
    %14 = llvm.call @omTensorGetStrides(%4) : (!llvm.ptr) -> !llvm.ptr
    %15 = llvm.getelementptr %13[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = llvm.insertvalue %16, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.getelementptr %14[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %19 = llvm.load %18 : !llvm.ptr -> i64
    %20 = llvm.insertvalue %19, %17[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.insertvalue %22, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.getelementptr %14[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.insertvalue %25, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.getelementptr %13[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %28 = llvm.load %27 : !llvm.ptr -> i64
    %29 = llvm.insertvalue %28, %26[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.getelementptr %14[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %31 = llvm.load %30 : !llvm.ptr -> i64
    %32 = llvm.insertvalue %31, %29[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.getelementptr %13[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %34 = llvm.load %33 : !llvm.ptr -> i64
    %35 = llvm.insertvalue %34, %32[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.getelementptr %14[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %37 = llvm.load %36 : !llvm.ptr -> i64
    %38 = llvm.insertvalue %37, %35[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.store %38, %5 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_main_graph_oneshotbufferize(%2, %5) : (!llvm.ptr, !llvm.ptr) -> ()
    %39 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %40 = llvm.mlir.constant(1 : i64) : i64
    %41 = llvm.alloca %40 x !llvm.ptr : (i64) -> !llvm.ptr
    %42 = llvm.mlir.constant(4 : i64) : i64
    %43 = llvm.call @omTensorCreateUntyped(%42) : (i64) -> !llvm.ptr
    %44 = llvm.mlir.constant(1 : i64) : i64
    %45 = llvm.extractvalue %39[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.bitcast %45 : !llvm.ptr to !llvm.ptr
    %47 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.bitcast %47 : !llvm.ptr to !llvm.ptr
    llvm.call @omTensorSetDataPtr(%43, %44, %46, %48) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    %49 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @omTensorSetDataType(%43, %49) : (!llvm.ptr, i64) -> ()
    %50 = llvm.call @omTensorGetShape(%43) : (!llvm.ptr) -> !llvm.ptr
    %51 = llvm.call @omTensorGetStrides(%43) : (!llvm.ptr) -> !llvm.ptr
    %52 = llvm.extractvalue %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.getelementptr %50[0] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %52, %53 : i64, !llvm.ptr
    %54 = llvm.extractvalue %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.getelementptr %51[0] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %54, %55 : i64, !llvm.ptr
    %56 = llvm.extractvalue %39[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.getelementptr %50[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %56, %57 : i64, !llvm.ptr
    %58 = llvm.extractvalue %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.getelementptr %51[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %58, %59 : i64, !llvm.ptr
    %60 = llvm.extractvalue %39[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.getelementptr %50[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %60, %61 : i64, !llvm.ptr
    %62 = llvm.extractvalue %39[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.getelementptr %51[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %62, %63 : i64, !llvm.ptr
    %64 = llvm.extractvalue %39[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %65 = llvm.getelementptr %50[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %64, %65 : i64, !llvm.ptr
    %66 = llvm.extractvalue %39[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %67 = llvm.getelementptr %51[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %66, %67 : i64, !llvm.ptr
    %68 = llvm.getelementptr %41[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %43, %68 : !llvm.ptr, !llvm.ptr
    %69 = llvm.call @omTensorListCreate(%41, %40) : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %69 : !llvm.ptr
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @run_main_graph_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_oneshotbufferize() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %1 = llvm.mlir.addressof @_entry_point_0_oneshotbufferize : !llvm.ptr
    %2 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
    %3 = llvm.insertvalue %2, %0[0] : !llvm.array<3 x ptr> 
    %4 = llvm.mlir.addressof @_entry_point_1_oneshotbufferize : !llvm.ptr
    %5 = llvm.bitcast %4 : !llvm.ptr to !llvm.ptr
    %6 = llvm.insertvalue %5, %3[1] : !llvm.array<3 x ptr> 
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = llvm.insertvalue %7, %6[2] : !llvm.array<3 x ptr> 
    llvm.return %8 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.icmp "ne" %arg0, %0 : !llvm.ptr
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = llvm.getelementptr %arg0[0] : (!llvm.ptr) -> !llvm.ptr, i64
    %3 = llvm.mlir.constant(2 : i64) : i64
    llvm.store %3, %2 : i64, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %4 = llvm.mlir.addressof @_entry_point_arrays_oneshotbufferize : !llvm.ptr
    %5 = llvm.bitcast %4 : !llvm.ptr to !llvm.ptr
    llvm.return %5 : !llvm.ptr
  }
  llvm.func @omQueryEntryPoints(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omQueryEntryPoints_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.addressof @_entry_point_0_oneshotbufferize : !llvm.ptr
    %2 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
    %3 = llvm.mlir.constant(15 : i64) : i64
    %4 = llvm.call @strncmp(%arg0, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %5 = llvm.icmp "eq" %4, %0 : i32
    llvm.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %6 = llvm.mlir.addressof @_entry_point_0_in_sig_oneshotbufferize : !llvm.ptr
    %7 = llvm.bitcast %6 : !llvm.ptr to !llvm.ptr
    llvm.return %7 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %8 = llvm.mlir.addressof @_entry_point_1_oneshotbufferize : !llvm.ptr
    %9 = llvm.bitcast %8 : !llvm.ptr to !llvm.ptr
    %10 = llvm.mlir.constant(32 : i64) : i64
    %11 = llvm.call @strncmp(%arg0, %9, %10) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %12 = llvm.icmp "eq" %11, %0 : i32
    llvm.cond_br %12, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %13 = llvm.mlir.addressof @_entry_point_1_in_sig_oneshotbufferize : !llvm.ptr
    %14 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr
    llvm.return %14 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    %15 = llvm.mlir.zero : !llvm.ptr
    llvm.return %15 : !llvm.ptr
  }
  llvm.func @omInputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omInputSignature_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.addressof @_entry_point_0_oneshotbufferize : !llvm.ptr
    %2 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
    %3 = llvm.mlir.constant(15 : i64) : i64
    %4 = llvm.call @strncmp(%arg0, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %5 = llvm.icmp "eq" %4, %0 : i32
    llvm.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %6 = llvm.mlir.addressof @_entry_point_0_out_sig_oneshotbufferize : !llvm.ptr
    %7 = llvm.bitcast %6 : !llvm.ptr to !llvm.ptr
    llvm.return %7 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %8 = llvm.mlir.addressof @_entry_point_1_oneshotbufferize : !llvm.ptr
    %9 = llvm.bitcast %8 : !llvm.ptr to !llvm.ptr
    %10 = llvm.mlir.constant(32 : i64) : i64
    %11 = llvm.call @strncmp(%arg0, %9, %10) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %12 = llvm.icmp "eq" %11, %0 : i32
    llvm.cond_br %12, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %13 = llvm.mlir.addressof @_entry_point_1_out_sig_oneshotbufferize : !llvm.ptr
    %14 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr
    llvm.return %14 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    %15 = llvm.mlir.zero : !llvm.ptr
    llvm.return %15 : !llvm.ptr
  }
  llvm.func @omOutputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omOutputSignature_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}

