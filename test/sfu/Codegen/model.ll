; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_entry_point_1_act_test_quant_sym = constant [34 x i8] c"run_main_graph_act_test_quant_sym\00"
@_entry_point_1_in_sig_act_test_quant_sym = constant [66 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00"
@_entry_point_1_out_sig_act_test_quant_sym = constant [66 x i8] c"[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00"
@_entry_point_0_act_test_quant_sym = constant [15 x i8] c"run_main_graph\00"
@_entry_point_0_in_sig_act_test_quant_sym = constant [66 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00"
@_entry_point_0_out_sig_act_test_quant_sym = constant [66 x i8] c"[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00"
@constant_4_act_test_quant_sym = internal constant [1 x float] [float 0x3F80204080000000], align 16
@constant_3_act_test_quant_sym = internal constant [1 x float] [float 0x3F97549840000000], align 16
@constant_2_act_test_quant_sym = internal constant [1 x i8] zeroinitializer, align 16
@constant_1_act_test_quant_sym = internal constant [32 x float] [float 0xBFC660A300000000, float 0x3FB4599780000000, float 0xBF997B3360000000, float 0xBFBBFE9100000000, float 0xBFB3C60C60000000, float 0xBFA2864E60000000, float 0x3FCC0BFD60000000, float 0x3FC80AD680000000, float 0x3FCF5E6E40000000, float 0xBF97DEA960000000, float 0x3FA314A860000000, float 0xBFAA28B5A0000000, float 0x3FA6E94520000000, float 0xBF60CD16E0000000, float 0xBFB5270D40000000, float 0xBFC42CD2E0000000, float 0x3FB9D792E0000000, float 0x3FBB51F500000000, float 0x3F9135CE60000000, float 0xBFB5C22A60000000, float 0xBF9CBE7E60000000, float 0xBF8589FD20000000, float 0x3FC71F51C0000000, float 0xBF8CA9BD40000000, float 0xBFA1F1E7A0000000, float 0x3F9A198E00000000, float 0xBFADC0F160000000, float 0xBFBF032A00000000, float 0x3FB05FBE60000000, float 0xBFA43DE7E0000000, float 0xBFC1FA1420000000, float 0xBFA9C22560000000], align 16
@constant_0_act_test_quant_sym = internal constant [32 x float] [float 0xBF501836E0000000, float 0xBFB7A6A360000000, float 0x3F9F9CBC60000000, float 0x3FC0136680000000, float 0xBFA8A921E0000000, float 0x3FA9383480000000, float 0xBFAB13D6E0000000, float 0x3FC8771D60000000, float 0xBFBA3B1B40000000, float 0x3F84082F20000000, float 0xBFB4EF6660000000, float 0x3F813C9460000000, float 0xBFB076C680000000, float 0xBFC016A960000000, float 0xBFA8A17440000000, float 0xBFBDA37DE0000000, float 0xBFC009A0E0000000, float 0x3FB38E0BC0000000, float 0x3FA1210E60000000, float 0xBF9962A0C0000000, float 0x3FB3BD8480000000, float 0x3FBA73FD60000000, float 0xBFC09288C0000000, float 0x3F75B412A0000000, float 0x3F874F5CC0000000, float 0xBFA1018140000000, float 0x3FB0292860000000, float 0x3F9935DFE0000000, float 0xBFBFABF940000000, float 0xBFAF098400000000, float 0x3FAEDA8600000000, float 0xBFB41BB3A0000000], align 16
@_entry_point_arrays_act_test_quant_sym = internal constant [3 x ptr] [ptr @_entry_point_0_act_test_quant_sym, ptr @_entry_point_1_act_test_quant_sym, ptr null]

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

declare void @npu_destroy()

declare void @free(ptr)

declare void @npu_dma_mvout(ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)

declare void @npu_sfu_run(i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16)

declare void @npu_dma_mvin(ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)

declare void @npu_mem_free(ptr)

declare ptr @npu_mem_alloc(i64)

declare ptr @malloc(i64)

declare i32 @npu_init()

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_graph_act_test_quant_sym(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = call i32 @npu_init()
  %9 = call ptr @malloc(i64 144)
  %10 = ptrtoint ptr %9 to i64
  %11 = add i64 %10, 15
  %12 = urem i64 %11, 16
  %13 = sub i64 %11, %12
  %14 = inttoptr i64 %13 to ptr
  br label %15

15:                                               ; preds = %18, %7
  %16 = phi i64 [ %27, %18 ], [ 0, %7 ]
  %17 = icmp slt i64 %16, 32
  br i1 %17, label %18, label %28

18:                                               ; preds = %15
  %19 = add nuw nsw i64 0, %16
  %20 = getelementptr inbounds nuw float, ptr %1, i64 %19
  %21 = load float, ptr %20, align 4
  %22 = getelementptr inbounds nuw float, ptr @constant_0_act_test_quant_sym, i64 %16
  %23 = load float, ptr %22, align 4
  %24 = fadd float %21, %23
  %25 = add nuw nsw i64 0, %16
  %26 = getelementptr inbounds nuw float, ptr %14, i64 %25
  store float %24, ptr %26, align 4
  %27 = add i64 %16, 1
  br label %15

28:                                               ; preds = %15
  %29 = call ptr @malloc(i64 128)
  %30 = call ptr @npu_mem_alloc(i64 32)
  %31 = load float, ptr @constant_3_act_test_quant_sym, align 4
  %32 = load i8, ptr @constant_2_act_test_quant_sym, align 1
  %33 = sext i8 %32 to i32
  %34 = sitofp i32 %33 to float
  %35 = call ptr @npu_mem_alloc(i64 8)
  store i64 32, ptr %35, align 8
  call void @npu_mem_free(ptr %35)
  %36 = call ptr @npu_mem_alloc(i64 8)
  store i64 32, ptr %36, align 8
  call void @npu_mem_free(ptr %36)
  br label %37

37:                                               ; preds = %40, %28
  %38 = phi i64 [ %63, %40 ], [ 0, %28 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  %41 = getelementptr inbounds nuw float, ptr %14, i64 %38
  %42 = load float, ptr %41, align 4
  %43 = fdiv float %42, %31
  %44 = call float @llvm.floor.f32(float %43)
  %45 = fsub float %43, %44
  %46 = fcmp ogt float %45, 5.000000e-01
  %47 = fadd float %44, 1.000000e+00
  %48 = select i1 %46, float %47, float %44
  %49 = fmul float %44, 5.000000e-01
  %50 = call float @llvm.floor.f32(float %49)
  %51 = fmul float %50, 2.000000e+00
  %52 = fsub float %44, %51
  %53 = fcmp oeq float %52, 1.000000e+00
  %54 = select i1 %53, float %47, float %44
  %55 = fcmp oeq float %45, 5.000000e-01
  %56 = select i1 %55, float %54, float %48
  %57 = fadd float %56, %34
  %58 = call float @llvm.maxnum.f32(float %57, float -1.280000e+02)
  %59 = call float @llvm.minnum.f32(float %58, float 1.270000e+02)
  %60 = fptosi float %59 to i32
  %61 = trunc i32 %60 to i8
  %62 = getelementptr inbounds nuw i8, ptr %30, i64 %38
  store i8 %61, ptr %62, align 1
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  %65 = call ptr @npu_mem_alloc(i64 32)
  call void @npu_dma_mvin(ptr %30, i32 0, i16 32, i16 1, i16 32, i16 32, i8 1, i8 0, i8 0, i1 false, i32 0, i16 0, i16 0)
  call void @npu_mem_free(ptr %30)
  call void @npu_sfu_run(i8 2, i8 8, i1 true, i32 0, i16 32, i16 1, i32 0, i32 0, i16 0, i16 23890, i16 -20, i16 22131, i16 -9)
  call void @npu_dma_mvout(ptr %65, i32 0, i16 32, i16 1, i16 32, i16 32, i8 1, i8 0, i8 0, i1 false, i32 0, i16 0, i16 0)
  %66 = call ptr @npu_mem_alloc(i64 32)
  call void @npu_dma_mvin(ptr %65, i32 0, i16 32, i16 1, i16 32, i16 32, i8 1, i8 0, i8 0, i1 false, i32 0, i16 0, i16 0)
  call void @npu_mem_free(ptr %65)
  call void @npu_sfu_run(i8 1, i8 8, i1 true, i32 0, i16 32, i16 1, i32 0, i32 0, i16 0, i16 24259, i16 -20, i16 22167, i16 -9)
  call void @npu_dma_mvout(ptr %66, i32 0, i16 32, i16 1, i16 32, i16 32, i8 1, i8 0, i8 0, i1 false, i32 0, i16 0, i16 0)
  %67 = call ptr @npu_mem_alloc(i64 32)
  call void @npu_dma_mvin(ptr %66, i32 0, i16 32, i16 1, i16 32, i16 32, i8 1, i8 0, i8 0, i1 false, i32 0, i16 0, i16 0)
  call void @npu_mem_free(ptr %66)
  call void @npu_sfu_run(i8 0, i8 8, i1 true, i32 0, i16 32, i16 1, i32 0, i32 0, i16 0, i16 24219, i16 -20, i16 32512, i16 -8)
  call void @npu_dma_mvout(ptr %67, i32 0, i16 32, i16 1, i16 32, i16 32, i8 1, i8 0, i8 0, i1 false, i32 0, i16 0, i16 0)
  br label %68

68:                                               ; preds = %71, %64
  %69 = phi i64 [ %85, %71 ], [ 0, %64 ]
  %70 = icmp slt i64 %69, 32
  br i1 %70, label %71, label %86

71:                                               ; preds = %68
  %72 = add nuw nsw i64 0, %69
  %73 = getelementptr inbounds nuw i8, ptr %67, i64 %72
  %74 = load i8, ptr %73, align 1
  %75 = load float, ptr @constant_4_act_test_quant_sym, align 4
  %76 = load i8, ptr @constant_2_act_test_quant_sym, align 1
  %77 = sext i8 %74 to i32
  %78 = sitofp i32 %77 to float
  %79 = sext i8 %76 to i32
  %80 = sitofp i32 %79 to float
  %81 = fsub float %78, %80
  %82 = fmul float %81, %75
  %83 = add nuw nsw i64 0, %69
  %84 = getelementptr inbounds nuw float, ptr %29, i64 %83
  store float %82, ptr %84, align 4
  %85 = add i64 %69, 1
  br label %68

86:                                               ; preds = %68
  call void @npu_mem_free(ptr %67)
  call void @free(ptr %9)
  %87 = call ptr @malloc(i64 144)
  %88 = ptrtoint ptr %87 to i64
  %89 = add i64 %88, 15
  %90 = urem i64 %89, 16
  %91 = sub i64 %89, %90
  %92 = inttoptr i64 %91 to ptr
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %87, 0
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, ptr %92, 1
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %94, i64 0, 2
  %96 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, i64 1, 3, 0
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, i64 32, 3, 1
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %97, i64 32, 4, 0
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, i64 1, 4, 1
  br label %100

100:                                              ; preds = %103, %86
  %101 = phi i64 [ %112, %103 ], [ 0, %86 ]
  %102 = icmp slt i64 %101, 32
  br i1 %102, label %103, label %113

103:                                              ; preds = %100
  %104 = add nuw nsw i64 0, %101
  %105 = getelementptr inbounds nuw float, ptr %29, i64 %104
  %106 = load float, ptr %105, align 4
  %107 = getelementptr inbounds nuw float, ptr @constant_1_act_test_quant_sym, i64 %101
  %108 = load float, ptr %107, align 4
  %109 = fadd float %106, %108
  %110 = add nuw nsw i64 0, %101
  %111 = getelementptr inbounds nuw float, ptr %92, i64 %110
  store float %109, ptr %111, align 4
  %112 = add i64 %101, 1
  br label %100

113:                                              ; preds = %100
  call void @free(ptr %29)
  call void @npu_destroy()
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %99
}

define void @_mlir_ciface_main_graph_act_test_quant_sym(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 1
  %11 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_graph_act_test_quant_sym(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, ptr %0, align 8
  ret void
}

define ptr @run_main_graph_act_test_quant_sym(ptr %0) {
  %2 = call ptr @omTensorListGetOmtArray(ptr %0)
  %3 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  %6 = call ptr @omTensorGetDataPtr(ptr %4)
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %6, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, i64 0, 2
  %10 = call ptr @omTensorGetShape(ptr %4)
  %11 = call ptr @omTensorGetStrides(ptr %4)
  %12 = load i64, ptr %10, align 8
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %12, 3, 0
  %14 = load i64, ptr %11, align 8
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %14, 4, 0
  %16 = getelementptr i64, ptr %10, i32 1
  %17 = load i64, ptr %16, align 8
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 %17, 3, 1
  %19 = getelementptr i64, ptr %11, i32 1
  %20 = load i64, ptr %19, align 8
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %20, 4, 1
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %5, align 8
  call void @_mlir_ciface_main_graph_act_test_quant_sym(ptr %3, ptr %5)
  %22 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %23 = alloca ptr, i64 1, align 8
  %24 = call ptr @omTensorCreateUntyped(i64 2)
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  call void @omTensorSetDataPtr(ptr %24, i64 1, ptr %25, ptr %26)
  call void @omTensorSetDataType(ptr %24, i64 1)
  %27 = call ptr @omTensorGetShape(ptr %24)
  %28 = call ptr @omTensorGetStrides(ptr %24)
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 3, 0
  store i64 %29, ptr %27, align 8
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 4, 0
  store i64 %30, ptr %28, align 8
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 3, 1
  %32 = getelementptr i64, ptr %27, i32 1
  store i64 %31, ptr %32, align 8
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 4, 1
  %34 = getelementptr i64, ptr %28, i32 1
  store i64 %33, ptr %34, align 8
  store ptr %24, ptr %23, align 8
  %35 = call ptr @omTensorListCreate(ptr %23, i64 1)
  ret ptr %35
}

define ptr @run_main_graph(ptr %0) {
  %2 = call ptr @run_main_graph_act_test_quant_sym(ptr %0)
  ret ptr %2
}

define ptr @omQueryEntryPoints_act_test_quant_sym(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  store i64 2, ptr %0, align 8
  br label %4

4:                                                ; preds = %3, %1
  ret ptr @_entry_point_arrays_act_test_quant_sym
}

define ptr @omQueryEntryPoints(ptr %0) {
  %2 = call ptr @omQueryEntryPoints_act_test_quant_sym(ptr %0)
  ret ptr %2
}

define ptr @omInputSignature_act_test_quant_sym(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_act_test_quant_sym, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  ret ptr @_entry_point_0_in_sig_act_test_quant_sym

5:                                                ; preds = %1
  %6 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_act_test_quant_sym, i64 34)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  ret ptr @_entry_point_1_in_sig_act_test_quant_sym

9:                                                ; preds = %5
  ret ptr null
}

define ptr @omInputSignature(ptr %0) {
  %2 = call ptr @omInputSignature_act_test_quant_sym(ptr %0)
  ret ptr %2
}

define ptr @omOutputSignature_act_test_quant_sym(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_act_test_quant_sym, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  ret ptr @_entry_point_0_out_sig_act_test_quant_sym

5:                                                ; preds = %1
  %6 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_act_test_quant_sym, i64 34)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  ret ptr @_entry_point_1_out_sig_act_test_quant_sym

9:                                                ; preds = %5
  ret ptr null
}

define ptr @omOutputSignature(ptr %0) {
  %2 = call ptr @omOutputSignature_act_test_quant_sym(ptr %0)
  ret ptr %2
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.floor.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minnum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
