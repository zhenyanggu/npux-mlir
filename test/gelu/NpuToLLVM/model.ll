; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-linux-gnueabihf"

@_entry_point_1_oneshotbufferize = constant [32 x i8] c"run_main_graph_oneshotbufferize\00"
@_entry_point_1_in_sig_oneshotbufferize = constant [77 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 224 , 224] , \22name\22 : \22input\22 }\0A\0A]\00"
@_entry_point_1_out_sig_oneshotbufferize = constant [78 x i8] c"[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 224 , 224] , \22name\22 : \22output\22 }\0A\0A]\00"
@_entry_point_0_oneshotbufferize = constant [15 x i8] c"run_main_graph\00"
@_entry_point_0_in_sig_oneshotbufferize = constant [77 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 224 , 224] , \22name\22 : \22input\22 }\0A\0A]\00"
@_entry_point_0_out_sig_oneshotbufferize = constant [78 x i8] c"[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 224 , 224] , \22name\22 : \22output\22 }\0A\0A]\00"
@constant_1_oneshotbufferize = internal constant [16 x float] [float 0xBF6FDA1C40000000, float 0x3FC2A0BC60000000, float 0x3F9818B920000000, float 0x3FC2E6BD60000000, float 0xBFA182AFA0000000, float 0x3FAE982A60000000, float 0x3FA79B8680000000, float 0x3FB1BE6E60000000, float 0x3FBC37C4E0000000, float 0xBFC8403E00000000, float 0x3FC7D940E0000000, float 0xBF8BF63DC0000000, float 0xBFB2891F80000000, float 0x3FABBD7D60000000, float 0xBFB2B8E680000000, float 0xBFA2399A80000000], align 16
@constant_0_oneshotbufferize = internal constant [1728 x i8] c"\A1@\86=n\88\E9\BC\9EY0\BC\F5\1F\FF=\EA\F1L=\AD\C7,\BE\BD\0A\BB\BC\95^#\BC:\93Z\BD)\1B6\BDQ\EA\10\BC\19\FC\EC\BD\93\18*=\91\CB\01\BE;\0C\00>O\BF\96\BD\ED\9D\94\BD\ECj<\BD\82\02>\BDb\F1\0F>\0Cv\12\BE\AD\C9\88\BD\E1?\F8\BD\E8G\9C=P\FA\0D\BEd~=>\1CX5=\90\B5\1D\BE\1F\B11\BE\C3p\A2\BD\8AL\CE;:H\A6\BD\F0\E1\96=\16\A6\B8=\C3\D9D\BEx\1A\97\BA\F3.\AE\BD\B83\C8\BD\15R3\BDB\A7\16\BEZ\11\92\BD\9E\0E\CE<\83\BA\AC=\B2W3>\08w\1F>\B7v\C4\BD\14\EA\0C>F\D89>tg\1F>\BF\94P=\D2\9F,\BEA$\FE<\09q\89\BD\F0\D9\AF=\E8\01(>Q\C5\FD\BD>\D8A=jW\EB<\AE\C9#>\A9v\8A<#|;\BE\DA\8B1=I\22\F2=0\EE\1E\BE\8BP\0F\BEM\88@>&\ED4\BEn\BE*>\82@\CA=\8A\D8\C4=\CF:B\BE@\0C\14\BE\01\FB\EC;\EC\88\EB\BA\03\93\85=\B3<\87=\AD/\87\BC\DC\B7+>\B0n%\BD\DE5\D2=\B8S2\BE\06&\98=\88\BA\EA=5\15B>\E5\09\90\BDu8\CB=\A6:6\BD\F0q\FD\BD\BE\A3\E7<<\18\8D=\BBF\E2\BD\BA\827>h\F0\18>u\9B\AC<\ECe\1B\BE\FD\86\A2\BC\11\DB#\BE\E1@4\BE\07\F9)>\0E\80\19>\BE\EE:>l\FD\D1\BDs\168>#s\F9=\A5\87\1C=\F1\98\81\BDE\83 \BE\95xJ=&1\A1\BB\03\0D\FD\BB\12\F5\B6\BD\0F\F2\BC\BDZD\12\BE\D9\F8\09<\C0\AD\03=\09]\09=\8D\D2\D8\BD\BA,\0C\BD\04_\B9\BC2\A1\B0=\FE\08\EB\BC\C6V\FC=\F8_x\BA\ED\89\9B\BCc\90\96\BD\98\EC\AF;\12\E9C\BC\06\F78\BD}\98\B2\BD\12] \BDRiL\BD\F5\9F0=\99\84\C8<\192\DA=Q\D2\03>\09K\A5=\15.\F5\BD\0E\13S\BDT-\FD\BD\86\AD)\BD\11\C2\9C=\03[[=R}\94\BD\9D{\CA<\D2A\B1\BD\F3\1BR=\E9u8=\C6H\0C=\DC\14B>\07\80\14\BE]\9ES\BD\C9\ECO=\99\9Ft=B\0F\0F\BE\D6 \9B=\A9\13\15\BEK\97\EC=NH\1B\BE\94\CB\18\BE\13^,\BENO\8A<=\1A\9E=m&\A2=U\16\17<\D0\8C\F3=\C8\A9\12>Rb\DC=\A9\0A\BE=\08\CF\01>\8F\1F\A9=\8F\EB%\BE%|\1A>\1C\AF3\BE\99\F9\BA\BD|\8C\FF\BD\AC\1A\C7\BD\F8N\1C>\8D\C9\16\BE\BA\E7\AE\BD\CF\A1A>dki\BD\FA\15\EA\BC\83\E3-\BD4\87\97=\C62\18\BE\F4\FB\14>6\F3\C3\BD\FA\8A;>\CA\A0\02>\8F{P\BD\B7f\B9<\17\ED\1C\BC\C9:C\BEM\F6\B3\BD\CF\06\A8\BDG\FF/\BE\D9\FC\E5\BD\DD_\A8\BC)\AE\FB=<\10\07>\A8\84\FF\BD`\D4\08>WC\0C>3n#>j\9A\94=\C3\8D\92<t\88\C0\BDE\DB\B2=\0Bt\EB\BD\8C\E4\FA=\DE/\8E<a=\0D\BE\91^\A8\BD\0Du9<\C4 B>\F5&<\BE7\C9\08\BE\E1\D2\10>\FA\9B\DD=\BD\9C\F8\BCG\9E8>Q\A1\B9\BDb\09\FA=\CA`\04\BEo\C6\06\BE\F1\18 \BE\12\E9\EE\BD\D8\D8\9F=\A4\14\F3=!\0A\09\BE\B2\8B\CC\BCi\B6>>e\86\F9\BD\BE\9AX=\1C\1D\EC\BBP\AD\94\BDUE\BA\BC\16\DCC>\D2\EF\A3=b+&\BD\AC\C7\8C\BD$\E2X=\B4/\B6;\D978>d\D4H\BD\BF\F5\AD\BD\87I\CA\BC\E3{+<\BC/\D4\BD\91\E7S=Zl\9C\BDh}?>\F6s$\BEL]\97\BC\9A\86\95\BD4)\22\BE\E4\87\EF\BC\B2M\C9=\E5*\1A\BEk\FF*>4/7\BE)`\17\BD\94h?\BE\A5I[=\88\BEp=\91\83l=\E9\09-\BE<\EC3<\B2\9B<>K\11=\BE\CB~8>\B4\DD\09\BE\0F\91$\BD\02\B6\89=\19\811>\E7\80\92=\95Z\B4\BD\10\A3\03>>\DB\E0\BD\A0F\12\BD\A1\06(\BE\F1\F9:>:m4=\EC`H\BD\09\C6%=.\F5D=\A5&\0B>\BBA*>\91\80\11>\87\F5%>@\BBD=_\BDz=\E3M0>\83U\AE\BD\82\F8\D0=m~\C8\BD\0C3\B8\BDO\A9l=x?;>\8FU\0B>\88\7F)\BE\F4\199=\FAf5<\98l\A5<\A5Ui\BD\22\0C\ED\BB\C3>=\BEC\13\E2=\8F\C2.\BE\85\F9\02>\1A\A9<>O}\06>\D3a=\BE\\\7F\F0\BDI\F4r\BDK\A4\18=\8B\E3\A7=-4<\BE\84\E9\DA<\9C\AA4>q\F3%>\E2\82\8C\BD\AC\9F\14\BD\A4o\0F\BE\0D\07=>\10\0A\ED\BC\C5\0BB=[}\8B\BC\1B\150>h,\14\BE\E4\E8+\BE@\AD<>6\9A\AB=.\09\93=]\98\E0=+k\02\BE\10\03\A7\BD\EF\D0w\BDKu\95\BD\8A\B6\C1\BC\90\05>\BD.|\81\BB\F0\03\A5\BD\BA\0A\DD\BD\E6w\22\BE]\F5\02>\CBF9\BE\0B\87\C7\BC3\D69\BE\B3\1E\8A\BD\BCn3>\05P->\D4\87\EB=@p\1A\BE\D7i\D5\BD\9A\D9?\BD9[\96\BD\87\1A5=\FA&[=\F5\A2\A6\BD\D5\85\17\BE\9F\0BE=$.\CC=\10\E8\13\BE\F5\ED\9C\BC\1A\91B\BE\10\D5\EB=w\ED\BE<\18XS\BD\E4\9F\05\BE\F0\D4\F9;\C3s\BC=-\0B8=\809\A5=\A9i\C9\BD\E6i\07\BE`Y\FD\BD\84\12\1C\BE;hy\BD\E9\F0\AD<\98%\C7=e-8>X:\C0=\14_\C2<9Y\1B>\16\BF\C4\BD\EA\FF&\BE\15\06:>n\C9\E0\BD\D3\83\B4=\88\FD\9A=p.\C4\BD\88\C5b\BC\80[\FE=l]j\BD\DD]\10>P\03\16>\1F\86\A2\BD:O6\BE\18\DA\A4<\\\7FP\BD\BB\1D\A9\BD\A8\88\B5\BB\D3+1>\E7rf\BD`\9B\90\BD\80\0F{<\A0\220=\1BH\A9\BD\F5\F9u\BCq&s\BC\AC\FD4\BE\F9\A5(>\D7}\AA=,\90!\BE\01L\12>\C7?\1A<\87\AE\04\BD#\C6-\BD\92\94(\BE\C7\9D\F9\BD[\D1\C7=\933\92=h\B5a<\07\DBA\BE;G >\17~\0A\BEN+\0C>&\B4C\BC\A1\F1\97;x0\A0\BD\077\1F\BE", align 16
@_entry_point_arrays_oneshotbufferize = internal constant [3 x ptr] [ptr @_entry_point_0_oneshotbufferize, ptr @_entry_point_1_oneshotbufferize, ptr null]

declare i32 @strncmp(ptr, ptr, i64)

declare void @omGetExternalConstantAddr(ptr, ptr, i64)

declare i1 @omMMapBinaryFile(ptr, ptr, i64, i64)

declare i64 @omTensorListGetSize(ptr)

declare void @omTensorPrint(ptr, ptr)

declare ptr @omTensorListGetOmtArray(ptr)

declare void @omTensorSetDataType(ptr, i64)

declare i64 @omTensorGetDataType(ptr)

declare ptr @omTensorGetStrides(ptr)

declare ptr @omTensorGetShape(ptr)

declare i64 @omTensorGetRank(ptr)

declare void @omTensorSetDataPtr(ptr, i64, ptr, ptr)

declare ptr @omTensorGetDataPtr(ptr)

declare void @omTensorDestroy(ptr)

declare ptr @omTensorCreateUntyped(i64)

declare ptr @omTensorListCreate(ptr, i64)

define private ptr @npu_mem_alloc(i64 %0) {
  %2 = call ptr @_mlir_ciface_npu_mem_alloc(i64 %0)
  ret ptr %2
}

declare ptr @_mlir_ciface_npu_mem_alloc(i64)

define private void @npu_dma_mvin(ptr %0, i32 %1, i16 %2, i16 %3, i16 %4, i16 %5, i8 %6, i8 %7, i8 %8, i1 %9, i32 %10, i16 %11, i16 %12) {
  call void @_mlir_ciface_npu_dma_mvin(ptr %0, i32 %1, i16 %2, i16 %3, i16 %4, i16 %5, i8 %6, i8 %7, i8 %8, i1 %9, i32 %10, i16 %11, i16 %12)
  ret void
}

declare void @_mlir_ciface_npu_dma_mvin(ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)

define private void @npu_sfu_run(i8 %0, i8 %1, i1 %2, i32 %3, i16 %4, i16 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12) {
  call void @_mlir_ciface_npu_sfu_run(i8 %0, i8 %1, i1 %2, i32 %3, i16 %4, i16 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12)
  ret void
}

declare void @_mlir_ciface_npu_sfu_run(i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16)

define private void @npu_dma_mvout(ptr %0, i32 %1, i16 %2, i16 %3, i16 %4, i16 %5, i8 %6, i8 %7, i8 %8, i1 %9, i32 %10, i16 %11, i16 %12) {
  call void @_mlir_ciface_npu_dma_mvout(ptr %0, i32 %1, i16 %2, i16 %3, i16 %4, i16 %5, i8 %6, i8 %7, i8 %8, i1 %9, i32 %10, i16 %11, i16 %12)
  ret void
}

declare void @_mlir_ciface_npu_dma_mvout(ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)

define private i32 @npu_init() {
  %1 = call i32 @_mlir_ciface_npu_init()
  ret i32 %1
}

declare i32 @_mlir_ciface_npu_init()

define { ptr, ptr, i64, [4 x i64], [4 x i64] } @main_graph_oneshotbufferize(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
  %12 = call i32 @npu_init()
  %13 = call ptr @npu_mem_alloc(i64 3211264)
  %14 = addrspacecast ptr %13 to ptr addrspace(2)
  br label %15

15:                                               ; preds = %111, %11
  %16 = phi i64 [ %112, %111 ], [ 0, %11 ]
  %17 = icmp slt i64 %16, 1
  br i1 %17, label %18, label %113

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %109, %18
  %20 = phi i64 [ %110, %109 ], [ 0, %18 ]
  %21 = icmp slt i64 %20, 1
  br i1 %21, label %22, label %111

22:                                               ; preds = %19
  br label %23

23:                                               ; preds = %107, %22
  %24 = phi i64 [ %108, %107 ], [ 0, %22 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %109

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %105, %26
  %28 = phi i64 [ %106, %105 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 224
  br i1 %29, label %30, label %107

30:                                               ; preds = %27
  br label %31

31:                                               ; preds = %91, %30
  %32 = phi i64 [ %104, %91 ], [ 0, %30 ]
  %33 = icmp slt i64 %32, 224
  br i1 %33, label %34, label %105

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %89, %34
  %36 = phi i64 [ %90, %89 ], [ 0, %34 ]
  %37 = phi float [ %47, %89 ], [ 0.000000e+00, %34 ]
  %38 = icmp slt i64 %36, 3
  br i1 %38, label %39, label %91

39:                                               ; preds = %35
  %40 = mul nsw i64 %28, -1
  %41 = add i64 %40, 1
  %42 = call i64 @llvm.smax.i64(i64 %41, i64 0)
  %43 = add i64 %40, 225
  %44 = call i64 @llvm.smin.i64(i64 %43, i64 3)
  br label %45

45:                                               ; preds = %87, %39
  %46 = phi i64 [ %88, %87 ], [ %42, %39 ]
  %47 = phi float [ %57, %87 ], [ %37, %39 ]
  %48 = icmp slt i64 %46, %44
  br i1 %48, label %49, label %89

49:                                               ; preds = %45
  %50 = mul nsw i64 %32, -1
  %51 = add i64 %50, 1
  %52 = call i64 @llvm.smax.i64(i64 %51, i64 0)
  %53 = add i64 %50, 225
  %54 = call i64 @llvm.smin.i64(i64 %53, i64 3)
  br label %55

55:                                               ; preds = %59, %49
  %56 = phi i64 [ %86, %59 ], [ %52, %49 ]
  %57 = phi float [ %85, %59 ], [ %47, %49 ]
  %58 = icmp slt i64 %56, %54
  br i1 %58, label %59, label %87

59:                                               ; preds = %55
  %60 = mul nsw i64 %20, 3
  %61 = add i64 %36, %60
  %62 = add i64 %46, %28
  %63 = add i64 %62, -1
  %64 = add i64 %56, %32
  %65 = add i64 %64, -1
  %66 = mul nuw nsw i64 %16, 150528
  %67 = mul nuw nsw i64 %61, 50176
  %68 = add nuw nsw i64 %66, %67
  %69 = mul nuw nsw i64 %63, 224
  %70 = add nuw nsw i64 %68, %69
  %71 = add nuw nsw i64 %70, %65
  %72 = getelementptr inbounds nuw float, ptr %1, i64 %71
  %73 = load float, ptr %72, align 4
  %74 = mul nsw i64 %20, 16
  %75 = add i64 %74, %24
  %76 = mul nuw nsw i64 %75, 27
  %77 = mul nuw nsw i64 %36, 9
  %78 = add nuw nsw i64 %76, %77
  %79 = mul nuw nsw i64 %46, 3
  %80 = add nuw nsw i64 %78, %79
  %81 = add nuw nsw i64 %80, %56
  %82 = getelementptr inbounds nuw float, ptr @constant_0_oneshotbufferize, i64 %81
  %83 = load float, ptr %82, align 4
  %84 = fmul float %73, %83
  %85 = fadd float %57, %84
  %86 = add i64 %56, 1
  br label %55

87:                                               ; preds = %55
  %88 = add i64 %46, 1
  br label %45

89:                                               ; preds = %45
  %90 = add i64 %36, 1
  br label %35

91:                                               ; preds = %35
  %92 = mul nsw i64 %20, 16
  %93 = add i64 %92, %24
  %94 = getelementptr inbounds nuw float, ptr @constant_1_oneshotbufferize, i64 %93
  %95 = load float, ptr %94, align 4
  %96 = fadd float %37, %95
  %97 = mul nuw nsw i64 %16, 802816
  %98 = mul nuw nsw i64 %93, 50176
  %99 = add nuw nsw i64 %97, %98
  %100 = mul nuw nsw i64 %28, 224
  %101 = add nuw nsw i64 %99, %100
  %102 = add nuw nsw i64 %101, %32
  %103 = getelementptr inbounds nuw float, ptr addrspace(2) %14, i64 %102
  store float %96, ptr addrspace(2) %103, align 4
  %104 = add i64 %32, 1
  br label %31

105:                                              ; preds = %31
  %106 = add i64 %28, 1
  br label %27

107:                                              ; preds = %27
  %108 = add i64 %24, 1
  br label %23

109:                                              ; preds = %23
  %110 = add i64 %20, 1
  br label %19

111:                                              ; preds = %19
  %112 = add i64 %16, 1
  br label %15

113:                                              ; preds = %15
  %114 = call ptr @npu_mem_alloc(i64 3211264)
  br label %115

115:                                              ; preds = %129, %113
  %116 = phi i64 [ %130, %129 ], [ 0, %113 ]
  %117 = icmp slt i64 %116, 224
  br i1 %117, label %118, label %131

118:                                              ; preds = %115
  br label %119

119:                                              ; preds = %122, %118
  %120 = phi i64 [ %128, %122 ], [ 0, %118 ]
  %121 = icmp slt i64 %120, 224
  br i1 %121, label %122, label %129

122:                                              ; preds = %119
  %123 = mul nsw i64 %116, 224
  %124 = add i64 %123, %120
  %125 = mul i64 %124, 4
  %126 = getelementptr i8, ptr %13, i64 %125
  call void @npu_dma_mvin(ptr %126, i32 0, i16 31, i16 31, i16 32, i16 224, i8 1, i8 0, i8 0, i1 false, i32 0, i16 1, i16 0)
  call void @npu_sfu_run(i8 1, i8 2, i1 false, i32 0, i16 31, i16 31, i32 65536, i32 0, i16 0, i16 1, i16 0, i16 1, i16 0)
  %127 = getelementptr i8, ptr %114, i64 %125
  call void @npu_dma_mvout(ptr %127, i32 65536, i16 31, i16 31, i16 32, i16 224, i8 1, i8 0, i8 0, i1 false, i32 0, i16 1, i16 0)
  %128 = add i64 %120, 32
  br label %119

129:                                              ; preds = %119
  %130 = add i64 %116, 32
  br label %115

131:                                              ; preds = %115
  %132 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } poison, ptr %114, 0
  %133 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %132, ptr %114, 1
  %134 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %133, i64 0, 2
  %135 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %134, i64 1, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %135, i64 802816, 4, 0
  %137 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %136, i64 16, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %137, i64 50176, 4, 1
  %139 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %138, i64 224, 3, 2
  %140 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %139, i64 224, 4, 2
  %141 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %140, i64 224, 3, 3
  %142 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %141, i64 1, 4, 3
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %142
}

define void @_mlir_ciface_main_graph_oneshotbufferize(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 2
  %10 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 3
  %11 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 2
  %14 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 3
  %15 = call { ptr, ptr, i64, [4 x i64], [4 x i64] } @main_graph_oneshotbufferize(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14)
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, ptr %0, align 8
  ret void
}

define ptr @run_main_graph_oneshotbufferize(ptr %0) {
  %2 = call ptr @omTensorListGetOmtArray(ptr %0)
  %3 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  %4 = load ptr, ptr %2, align 4
  %5 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  %6 = call ptr @omTensorGetDataPtr(ptr %4)
  %7 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %6, 0
  %8 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %8, i64 0, 2
  %10 = call ptr @omTensorGetShape(ptr %4)
  %11 = call ptr @omTensorGetStrides(ptr %4)
  %12 = load i64, ptr %10, align 8
  %13 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %9, i64 %12, 3, 0
  %14 = load i64, ptr %11, align 8
  %15 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %13, i64 %14, 4, 0
  %16 = getelementptr i64, ptr %10, i32 1
  %17 = load i64, ptr %16, align 8
  %18 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, i64 %17, 3, 1
  %19 = getelementptr i64, ptr %11, i32 1
  %20 = load i64, ptr %19, align 8
  %21 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %18, i64 %20, 4, 1
  %22 = getelementptr i64, ptr %10, i32 2
  %23 = load i64, ptr %22, align 8
  %24 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %21, i64 %23, 3, 2
  %25 = getelementptr i64, ptr %11, i32 2
  %26 = load i64, ptr %25, align 8
  %27 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %24, i64 %26, 4, 2
  %28 = getelementptr i64, ptr %10, i32 3
  %29 = load i64, ptr %28, align 8
  %30 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %27, i64 %29, 3, 3
  %31 = getelementptr i64, ptr %11, i32 3
  %32 = load i64, ptr %31, align 8
  %33 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %30, i64 %32, 4, 3
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, ptr %5, align 8
  call void @_mlir_ciface_main_graph_oneshotbufferize(ptr %3, ptr %5)
  %34 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %3, align 8
  %35 = alloca ptr, i64 1, align 4
  %36 = call ptr @omTensorCreateUntyped(i64 4)
  %37 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 0
  %38 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 1
  call void @omTensorSetDataPtr(ptr %36, i64 1, ptr %37, ptr %38)
  call void @omTensorSetDataType(ptr %36, i64 1)
  %39 = call ptr @omTensorGetShape(ptr %36)
  %40 = call ptr @omTensorGetStrides(ptr %36)
  %41 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 3, 0
  store i64 %41, ptr %39, align 8
  %42 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 4, 0
  store i64 %42, ptr %40, align 8
  %43 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 3, 1
  %44 = getelementptr i64, ptr %39, i32 1
  store i64 %43, ptr %44, align 8
  %45 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 4, 1
  %46 = getelementptr i64, ptr %40, i32 1
  store i64 %45, ptr %46, align 8
  %47 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 3, 2
  %48 = getelementptr i64, ptr %39, i32 2
  store i64 %47, ptr %48, align 8
  %49 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 4, 2
  %50 = getelementptr i64, ptr %40, i32 2
  store i64 %49, ptr %50, align 8
  %51 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 3, 3
  %52 = getelementptr i64, ptr %39, i32 3
  store i64 %51, ptr %52, align 8
  %53 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, 4, 3
  %54 = getelementptr i64, ptr %40, i32 3
  store i64 %53, ptr %54, align 8
  store ptr %36, ptr %35, align 4
  %55 = call ptr @omTensorListCreate(ptr %35, i64 1)
  ret ptr %55
}

define ptr @run_main_graph(ptr %0) {
  %2 = call ptr @run_main_graph_oneshotbufferize(ptr %0)
  ret ptr %2
}

define ptr @omQueryEntryPoints_oneshotbufferize(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  store i64 2, ptr %0, align 8
  br label %4

4:                                                ; preds = %3, %1
  ret ptr @_entry_point_arrays_oneshotbufferize
}

define ptr @omQueryEntryPoints(ptr %0) {
  %2 = call ptr @omQueryEntryPoints_oneshotbufferize(ptr %0)
  ret ptr %2
}

define ptr @omInputSignature_oneshotbufferize(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_oneshotbufferize, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %9, %6, %1
  %5 = phi ptr [ %10, %9 ], [ @_entry_point_1_in_sig_oneshotbufferize, %6 ], [ @_entry_point_0_in_sig_oneshotbufferize, %1 ]
  ret ptr %5

6:                                                ; preds = %1
  %7 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_oneshotbufferize, i64 32)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %4, label %9

9:                                                ; preds = %6
  %10 = phi ptr [ null, %6 ]
  br label %4
}

define ptr @omInputSignature(ptr %0) {
  %2 = call ptr @omInputSignature_oneshotbufferize(ptr %0)
  ret ptr %2
}

define ptr @omOutputSignature_oneshotbufferize(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_oneshotbufferize, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %9, %6, %1
  %5 = phi ptr [ %10, %9 ], [ @_entry_point_1_out_sig_oneshotbufferize, %6 ], [ @_entry_point_0_out_sig_oneshotbufferize, %1 ]
  ret ptr %5

6:                                                ; preds = %1
  %7 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_oneshotbufferize, i64 32)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %4, label %9

9:                                                ; preds = %6
  %10 = phi ptr [ null, %6 ]
  br label %4
}

define ptr @omOutputSignature(ptr %0) {
  %2 = call ptr @omOutputSignature_oneshotbufferize(ptr %0)
  ret ptr %2
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
