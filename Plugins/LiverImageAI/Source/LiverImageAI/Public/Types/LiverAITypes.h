#pragma once
#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "LiverAITypes.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogLiverAI, Log, All);

USTRUCT(BlueprintType)
struct LIVERIMAGEAI_API FLiverAnalysisRequest
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString MRIFilePath;

    UPROPERTY(BlueprintReadOnly)
    FString LiverModelPath;

    UPROPERTY(BlueprintReadOnly)
    FString VesselModelPath;

    UPROPERTY(BlueprintReadOnly)
    FString TumorModelPath;

    UPROPERTY(BlueprintReadOnly)
    FString RequestId;

    UPROPERTY(BlueprintReadOnly)
    bool bGenerate3DMesh;

    FLiverAnalysisRequest()
    {
        MRIFilePath = TEXT("");
        LiverModelPath = TEXT("");
        VesselModelPath = TEXT("");
        TumorModelPath = TEXT("");
        RequestId = TEXT("");
        bGenerate3DMesh = true;
    }
};

USTRUCT(BlueprintType)
struct LIVERIMAGEAI_API FAnalysisProgress
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString Status;

    UPROPERTY(BlueprintReadOnly)
    float Progress;

    UPROPERTY(BlueprintReadOnly)
    FString Message;

    UPROPERTY(BlueprintReadOnly)
    FString RequestId;

    FAnalysisProgress()
    {
        Status = TEXT("Ready");
        Progress = 0.0f;
        Message = TEXT("Ready");
        RequestId = TEXT("");
    }
};

USTRUCT(BlueprintType)
struct LIVERIMAGEAI_API FOrganVolumeStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString OrganName;

    UPROPERTY(BlueprintReadOnly)
    float VolumeML;

    UPROPERTY(BlueprintReadOnly)
    int32 VoxelCount;

    UPROPERTY(BlueprintReadOnly)
    int32 NumComponents;

    FOrganVolumeStats()
    {
        OrganName = TEXT("");
        VolumeML = 0.0f;
        VoxelCount = 0;
        NumComponents = 0;
    }
};

USTRUCT(BlueprintType)
struct LIVERIMAGEAI_API FMeshData
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString OrganName;

    UPROPERTY(BlueprintReadOnly)
    TArray<FVector> Vertices;

    UPROPERTY(BlueprintReadOnly)
    TArray<int32> Triangles;

    UPROPERTY(BlueprintReadOnly)
    FLinearColor Color;

    FMeshData()
    {
        OrganName = TEXT("");
        Color = FLinearColor::White;
    }
};

USTRUCT(BlueprintType)
struct LIVERIMAGEAI_API FLiverAnalysisResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString RequestId;

    UPROPERTY(BlueprintReadOnly)
    FString Timestamp;

    UPROPERTY(BlueprintReadOnly)
    bool bSuccess;

    UPROPERTY(BlueprintReadOnly)
    TArray<FOrganVolumeStats> OrganStats;

    UPROPERTY(BlueprintReadOnly)
    TArray<FMeshData> MeshData;

    UPROPERTY(BlueprintReadOnly)
    FString ErrorMessage;

    UPROPERTY(BlueprintReadOnly)
    FString ResultSummary;

    FLiverAnalysisResult()
    {
        RequestId = TEXT("");
        Timestamp = TEXT("");
        bSuccess = false;
        ErrorMessage = TEXT("");
        ResultSummary = TEXT("");
    }
};