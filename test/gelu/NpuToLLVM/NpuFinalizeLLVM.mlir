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
    %0 = llvm.mlir.constant(224 : i64) : i64
    %1 = llvm.mlir.constant(50176 : i64) : i64
    %2 = llvm.mlir.constant(16 : i64) : i64
    %3 = llvm.mlir.constant(802816 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.poison : !llvm.struct<(ptr<2>, ptr<2>, i64)>
    %7 = llvm.mlir.constant(32 : index) : i64
    %8 = llvm.mlir.constant(4 : index) : i64
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(31 : i16) : i16
    %11 = llvm.mlir.constant(32 : i16) : i16
    %12 = llvm.mlir.constant(224 : i16) : i16
    %13 = llvm.mlir.constant(1 : i8) : i8
    %14 = llvm.mlir.constant(0 : i8) : i8
    %15 = llvm.mlir.constant(false) : i1
    %16 = llvm.mlir.constant(1 : i16) : i16
    %17 = llvm.mlir.constant(0 : i16) : i16
    %18 = llvm.mlir.constant(65536 : i32) : i32
    %19 = llvm.mlir.constant(2 : i8) : i8
    %20 = llvm.mlir.constant(802816 : index) : i64
    %21 = llvm.mlir.constant(50176 : index) : i64
    %22 = llvm.mlir.constant(150528 : index) : i64
    %23 = llvm.mlir.constant(225 : index) : i64
    %24 = llvm.mlir.constant(-1 : index) : i64
    %25 = llvm.mlir.constant(224 : index) : i64
    %26 = llvm.mlir.addressof @constant_1_oneshotbufferize : !llvm.ptr
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(9 : index) : i64
    %29 = llvm.mlir.constant(3 : index) : i64
    %30 = llvm.mlir.constant(27 : index) : i64
    %31 = llvm.mlir.constant(16 : index) : i64
    %32 = llvm.mlir.constant(0 : index) : i64
    %33 = llvm.mlir.addressof @constant_0_oneshotbufferize : !llvm.ptr
    %34 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %35 = llvm.mlir.constant(3211264 : i64) : i64
    %36 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %37 = llvm.call @npu_init() : () -> i32
    %38 = llvm.call @npu_mem_alloc(%35) : (i64) -> !llvm.ptr
    %39 = llvm.addrspacecast %38 : !llvm.ptr to !llvm.ptr<2>
    llvm.br ^bb1(%32 : i64)
  ^bb1(%40: i64):  // 2 preds: ^bb0, ^bb23
    %41 = llvm.icmp "slt" %40, %27 : i64
    llvm.cond_br %41, ^bb2, ^bb24
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%32 : i64)
  ^bb3(%42: i64):  // 2 preds: ^bb2, ^bb22
    %43 = llvm.icmp "slt" %42, %27 : i64
    llvm.cond_br %43, ^bb4, ^bb23
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%32 : i64)
  ^bb5(%44: i64):  // 2 preds: ^bb4, ^bb21
    %45 = llvm.icmp "slt" %44, %31 : i64
    llvm.cond_br %45, ^bb6, ^bb22
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%32 : i64)
  ^bb7(%46: i64):  // 2 preds: ^bb6, ^bb20
    %47 = llvm.icmp "slt" %46, %25 : i64
    llvm.cond_br %47, ^bb8, ^bb21
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%32 : i64)
  ^bb9(%48: i64):  // 2 preds: ^bb8, ^bb19
    %49 = llvm.icmp "slt" %48, %25 : i64
    llvm.cond_br %49, ^bb10, ^bb20
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%32, %34 : i64, f32)
  ^bb11(%50: i64, %51: f32):  // 2 preds: ^bb10, ^bb18
    %52 = llvm.icmp "slt" %50, %29 : i64
    llvm.cond_br %52, ^bb12, ^bb19
  ^bb12:  // pred: ^bb11
    %53 = llvm.mul %46, %24 overflow<nsw> : i64
    %54 = llvm.add %53, %27 : i64
    %55 = llvm.intr.smax(%54, %32) : (i64, i64) -> i64
    %56 = llvm.mul %46, %24 overflow<nsw> : i64
    %57 = llvm.add %56, %23 : i64
    %58 = llvm.intr.smin(%57, %29) : (i64, i64) -> i64
    llvm.br ^bb13(%55, %51 : i64, f32)
  ^bb13(%59: i64, %60: f32):  // 2 preds: ^bb12, ^bb17
    %61 = llvm.icmp "slt" %59, %58 : i64
    llvm.cond_br %61, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %62 = llvm.mul %48, %24 overflow<nsw> : i64
    %63 = llvm.add %62, %27 : i64
    %64 = llvm.intr.smax(%63, %32) : (i64, i64) -> i64
    %65 = llvm.mul %48, %24 overflow<nsw> : i64
    %66 = llvm.add %65, %23 : i64
    %67 = llvm.intr.smin(%66, %29) : (i64, i64) -> i64
    llvm.br ^bb15(%64, %60 : i64, f32)
  ^bb15(%68: i64, %69: f32):  // 2 preds: ^bb14, ^bb16
    %70 = llvm.icmp "slt" %68, %67 : i64
    llvm.cond_br %70, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %71 = llvm.mul %42, %29 overflow<nsw> : i64
    %72 = llvm.add %50, %71 : i64
    %73 = llvm.add %59, %46 : i64
    %74 = llvm.add %73, %24 : i64
    %75 = llvm.add %68, %48 : i64
    %76 = llvm.add %75, %24 : i64
    %77 = llvm.mul %40, %22 overflow<nsw, nuw> : i64
    %78 = llvm.mul %72, %21 overflow<nsw, nuw> : i64
    %79 = llvm.add %77, %78 overflow<nsw, nuw> : i64
    %80 = llvm.mul %74, %25 overflow<nsw, nuw> : i64
    %81 = llvm.add %79, %80 overflow<nsw, nuw> : i64
    %82 = llvm.add %81, %76 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds|nuw %arg1[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.load %83 : !llvm.ptr -> f32
    %85 = llvm.mul %42, %31 overflow<nsw> : i64
    %86 = llvm.add %85, %44 : i64
    %87 = llvm.mul %86, %30 overflow<nsw, nuw> : i64
    %88 = llvm.mul %50, %28 overflow<nsw, nuw> : i64
    %89 = llvm.add %87, %88 overflow<nsw, nuw> : i64
    %90 = llvm.mul %59, %29 overflow<nsw, nuw> : i64
    %91 = llvm.add %89, %90 overflow<nsw, nuw> : i64
    %92 = llvm.add %91, %68 overflow<nsw, nuw> : i64
    %93 = llvm.getelementptr inbounds|nuw %33[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %94 = llvm.load %93 : !llvm.ptr -> f32
    %95 = llvm.fmul %84, %94 : f32
    %96 = llvm.fadd %69, %95 : f32
    %97 = llvm.add %68, %27 : i64
    llvm.br ^bb15(%97, %96 : i64, f32)
  ^bb17:  // pred: ^bb15
    %98 = llvm.add %59, %27 : i64
    llvm.br ^bb13(%98, %69 : i64, f32)
  ^bb18:  // pred: ^bb13
    %99 = llvm.add %50, %27 : i64
    llvm.br ^bb11(%99, %60 : i64, f32)
  ^bb19:  // pred: ^bb11
    %100 = llvm.mul %42, %31 overflow<nsw> : i64
    %101 = llvm.add %100, %44 : i64
    %102 = llvm.getelementptr inbounds|nuw %26[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.load %102 : !llvm.ptr -> f32
    %104 = llvm.fadd %51, %103 : f32
    %105 = llvm.mul %42, %31 overflow<nsw> : i64
    %106 = llvm.add %105, %44 : i64
    %107 = llvm.mul %40, %20 overflow<nsw, nuw> : i64
    %108 = llvm.mul %106, %21 overflow<nsw, nuw> : i64
    %109 = llvm.add %107, %108 overflow<nsw, nuw> : i64
    %110 = llvm.mul %46, %25 overflow<nsw, nuw> : i64
    %111 = llvm.add %109, %110 overflow<nsw, nuw> : i64
    %112 = llvm.add %111, %48 overflow<nsw, nuw> : i64
    %113 = llvm.getelementptr inbounds|nuw %39[%112] : (!llvm.ptr<2>, i64) -> !llvm.ptr<2>, f32
    llvm.store %104, %113 : f32, !llvm.ptr<2>
    %114 = llvm.add %48, %27 : i64
    llvm.br ^bb9(%114 : i64)
  ^bb20:  // pred: ^bb9
    %115 = llvm.add %46, %27 : i64
    llvm.br ^bb7(%115 : i64)
  ^bb21:  // pred: ^bb7
    %116 = llvm.add %44, %27 : i64
    llvm.br ^bb5(%116 : i64)
  ^bb22:  // pred: ^bb5
    %117 = llvm.add %42, %27 : i64
    llvm.br ^bb3(%117 : i64)
  ^bb23:  // pred: ^bb3
    %118 = llvm.add %40, %27 : i64
    llvm.br ^bb1(%118 : i64)
  ^bb24:  // pred: ^bb1
    %119 = llvm.call @npu_mem_alloc(%35) : (i64) -> !llvm.ptr
    %120 = llvm.addrspacecast %119 : !llvm.ptr to !llvm.ptr<2>
    llvm.br ^bb25(%32 : i64)
  ^bb25(%121: i64):  // 2 preds: ^bb24, ^bb29
    %122 = llvm.icmp "slt" %121, %25 : i64
    llvm.cond_br %122, ^bb26, ^bb30
  ^bb26:  // pred: ^bb25
    llvm.br ^bb27(%32 : i64)
  ^bb27(%123: i64):  // 2 preds: ^bb26, ^bb28
    %124 = llvm.icmp "slt" %123, %25 : i64
    llvm.cond_br %124, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    // [Manual Fix] 移除了 %125-%128 以及 %131 的无效转换
    // 原逻辑：MemRef 构建后再拆包。现逻辑：直接使用 %39 (Aligned Ptr)

    // 1. 计算 mvin 的偏移量 (Byte Offset)
    %129 = llvm.mul %121, %25 overflow<nsw> : i64
    %130 = llvm.add %129, %123 : i64
    %132 = llvm.mul %130, %8 : i64

    // 2. [关键修改] 指针转换与地址计算
    // 将 ptr<2> 转换为通用 ptr 以匹配函数签名
    %real_ptr_in = llvm.addrspacecast %39 : !llvm.ptr<2> to !llvm.ptr
    // 基于通用指针计算偏移地址 (i8 粒度)
    %133 = llvm.getelementptr %real_ptr_in[%132] : (!llvm.ptr, i64) -> !llvm.ptr, i8

    // 3. 调用 NPU DMA Input
    llvm.call @npu_dma_mvin(%133, %9, %10, %10, %11, %12, %13, %14, %14, %15, %9, %16, %17) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()

    // 4. 调用 NPU SFU (计算)
    llvm.call @npu_sfu_run(%13, %19, %15, %9, %10, %10, %18, %9, %17, %16, %17, %16, %17) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()

    // [Manual Fix] 移除了 %134-%137 以及 %140 的无效转换
    // 原逻辑：MemRef 构建后再拆包。现逻辑：直接使用 %120 (Aligned Ptr)

    // 5. 计算 mvout 的偏移量 (Byte Offset)
    %138 = llvm.mul %121, %25 overflow<nsw> : i64
    %139 = llvm.add %138, %123 : i64
    %141 = llvm.mul %139, %8 : i64

    // 6. [关键修改] 指针转换与地址计算
    // 将 ptr<2> 转换为通用 ptr
    %real_ptr_out = llvm.addrspacecast %120 : !llvm.ptr<2> to !llvm.ptr
    // 基于通用指针计算偏移地址 (i8 粒度)
    %142 = llvm.getelementptr %real_ptr_out[%141] : (!llvm.ptr, i64) -> !llvm.ptr, i8

    // 7. 调用 NPU DMA Output
    llvm.call @npu_dma_mvout(%142, %18, %10, %10, %11, %12, %13, %14, %14, %15, %9, %16, %17) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()

    // 8. 循环变量更新与跳转
    %143 = llvm.add %123, %7 : i64
    llvm.br ^bb27(%143 : i64)
  ^bb29:  // pred: ^bb27
    %144 = llvm.add %121, %7 : i64
    llvm.br ^bb25(%144 : i64)
  ^bb30:  // pred: ^bb25
    %145 = llvm.insertvalue %119, %36[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %146 = llvm.insertvalue %119, %145[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %147 = llvm.insertvalue %5, %146[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %148 = llvm.insertvalue %4, %147[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %149 = llvm.insertvalue %3, %148[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %150 = llvm.insertvalue %2, %149[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %151 = llvm.insertvalue %1, %150[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %152 = llvm.insertvalue %0, %151[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %153 = llvm.insertvalue %0, %152[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %154 = llvm.insertvalue %0, %153[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %155 = llvm.insertvalue %4, %154[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.return %155 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
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
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    %6 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    %8 = llvm.call @omTensorGetDataPtr(%6) : (!llvm.ptr) -> !llvm.ptr
    %9 = llvm.insertvalue %8, %2[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %1, %10[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.call @omTensorGetShape(%6) : (!llvm.ptr) -> !llvm.ptr
    %13 = llvm.call @omTensorGetStrides(%6) : (!llvm.ptr) -> !llvm.ptr
    %14 = llvm.load %12 : !llvm.ptr -> i64
    %15 = llvm.insertvalue %14, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.load %13 : !llvm.ptr -> i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %19 = llvm.load %18 : !llvm.ptr -> i64
    %20 = llvm.insertvalue %19, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.insertvalue %22, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.getelementptr %12[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.insertvalue %25, %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.getelementptr %13[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %28 = llvm.load %27 : !llvm.ptr -> i64
    %29 = llvm.insertvalue %28, %26[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.getelementptr %12[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %31 = llvm.load %30 : !llvm.ptr -> i64
    %32 = llvm.insertvalue %31, %29[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.getelementptr %13[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %34 = llvm.load %33 : !llvm.ptr -> i64
    %35 = llvm.insertvalue %34, %32[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.store %35, %7 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_main_graph_oneshotbufferize(%5, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    %36 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %37 = llvm.alloca %3 x !llvm.ptr : (i64) -> !llvm.ptr
    %38 = llvm.call @omTensorCreateUntyped(%0) : (i64) -> !llvm.ptr
    %39 = llvm.extractvalue %36[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @omTensorSetDataPtr(%38, %3, %39, %40) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%38, %3) : (!llvm.ptr, i64) -> ()
    %41 = llvm.call @omTensorGetShape(%38) : (!llvm.ptr) -> !llvm.ptr
    %42 = llvm.call @omTensorGetStrides(%38) : (!llvm.ptr) -> !llvm.ptr
    %43 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.store %43, %41 : i64, !llvm.ptr
    %44 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.store %44, %42 : i64, !llvm.ptr
    %45 = llvm.extractvalue %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.getelementptr %41[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %45, %46 : i64, !llvm.ptr
    %47 = llvm.extractvalue %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.getelementptr %42[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %47, %48 : i64, !llvm.ptr
    %49 = llvm.extractvalue %36[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.getelementptr %41[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %49, %50 : i64, !llvm.ptr
    %51 = llvm.extractvalue %36[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %52 = llvm.getelementptr %42[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %51, %52 : i64, !llvm.ptr
    %53 = llvm.extractvalue %36[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %54 = llvm.getelementptr %41[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %53, %54 : i64, !llvm.ptr
    %55 = llvm.extractvalue %36[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.getelementptr %42[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %55, %56 : i64, !llvm.ptr
    llvm.store %38, %37 : !llvm.ptr, !llvm.ptr
    %57 = llvm.call @omTensorListCreate(%37, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %57 : !llvm.ptr
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @run_main_graph_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_oneshotbufferize() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_oneshotbufferize : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @_entry_point_0_oneshotbufferize : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @_entry_point_arrays_oneshotbufferize : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "ne" %arg0, %2 : !llvm.ptr
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.store %1, %arg0 : i64, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omQueryEntryPoints(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omQueryEntryPoints_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_in_sig_oneshotbufferize : !llvm.ptr
    %2 = llvm.mlir.constant(32 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_oneshotbufferize : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_in_sig_oneshotbufferize : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_oneshotbufferize : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1(%4 : !llvm.ptr), ^bb2
  ^bb1(%10: !llvm.ptr):  // 3 preds: ^bb0, ^bb2, ^bb2
    llvm.return %10 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %11 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %12 = llvm.icmp "eq" %11, %6 : i32
    llvm.cond_br %12, ^bb1(%1 : !llvm.ptr), ^bb1(%0 : !llvm.ptr)
  }
  llvm.func @omInputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omInputSignature_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_oneshotbufferize(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_out_sig_oneshotbufferize : !llvm.ptr
    %2 = llvm.mlir.constant(32 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_oneshotbufferize : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_out_sig_oneshotbufferize : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_oneshotbufferize : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1(%4 : !llvm.ptr), ^bb2
  ^bb1(%10: !llvm.ptr):  // 3 preds: ^bb0, ^bb2, ^bb2
    llvm.return %10 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %11 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %12 = llvm.icmp "eq" %11, %6 : i32
    llvm.cond_br %12, ^bb1(%1 : !llvm.ptr), ^bb1(%0 : !llvm.ptr)
  }
  llvm.func @omOutputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omOutputSignature_oneshotbufferize(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}

