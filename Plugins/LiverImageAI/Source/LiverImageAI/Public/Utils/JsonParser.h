#pragma once
#include "Types/LiverAITypes.h"
#include "Dom/JsonObject.h"

class LIVERIMAGEAI_API FJsonParser
{
public:
    FLiverAnalysisResult ParseAnalysisResult(const FString& JsonString);
    FAnalysisProgress ParseProgressUpdate(const FString& JsonString);
    TArray<FMeshData> ParseMeshData(TSharedPtr<FJsonObject> JsonObject);
    TArray<FOrganVolumeStats> ParseOrganStats(TSharedPtr<FJsonObject> JsonObject);
    
private:
    TSharedPtr<FJsonObject> StringToJson(const FString& JsonString);
    FMeshData ParseSingleMesh(TSharedPtr<FJsonObject> MeshObject);
    FOrganVolumeStats ParseSingleOrganStats(TSharedPtr<FJsonObject> StatsObject);
};