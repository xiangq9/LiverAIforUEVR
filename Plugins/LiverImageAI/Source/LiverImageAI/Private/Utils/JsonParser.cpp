#include "Utils/JsonParser.h"
#include "Json.h"
#include "Types/LiverAITypes.h"

FLiverAnalysisResult FJsonParser::ParseAnalysisResult(const FString& JsonString)
{
    FLiverAnalysisResult Result;
    
    TSharedPtr<FJsonObject> JsonObject = StringToJson(JsonString);
    if (!JsonObject.IsValid())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Invalid JSON");
        return Result;
    }
    
    Result.bSuccess = JsonObject->GetBoolField(TEXT("success"));
    Result.RequestId = JsonObject->GetStringField(TEXT("request_id"));
    Result.ErrorMessage = JsonObject->GetStringField(TEXT("error_message"));
    Result.ResultSummary = JsonObject->GetStringField(TEXT("diagnostic_report"));
    Result.Timestamp = FDateTime::Now().ToString();
    
    // Parse organ statistics
    const TArray<TSharedPtr<FJsonValue>>* OrganStatsArray;
    if (JsonObject->TryGetArrayField(TEXT("organ_stats"), OrganStatsArray))
    {
        for (const auto& Value : *OrganStatsArray)
        {
            if (Value->Type == EJson::Object)
            {
                Result.OrganStats.Add(ParseSingleOrganStats(Value->AsObject()));
            }
        }
    }
    
    // Parse mesh data
    const TArray<TSharedPtr<FJsonValue>>* MeshDataArray;
    if (JsonObject->TryGetArrayField(TEXT("mesh_data"), MeshDataArray))
    {
        Result.MeshData = ParseMeshData(JsonObject);
    }
    
    return Result;
}

FAnalysisProgress FJsonParser::ParseProgressUpdate(const FString& JsonString)
{
    FAnalysisProgress Progress;
    
    TSharedPtr<FJsonObject> JsonObject = StringToJson(JsonString);
    if (!JsonObject.IsValid())
    {
        return Progress;
    }
    
    Progress.Status = JsonObject->GetStringField(TEXT("status"));
    Progress.Progress = JsonObject->GetNumberField(TEXT("progress"));
    Progress.Message = JsonObject->GetStringField(TEXT("message"));
    Progress.RequestId = JsonObject->GetStringField(TEXT("request_id"));
    
    return Progress;
}

TArray<FMeshData> FJsonParser::ParseMeshData(TSharedPtr<FJsonObject> JsonObject)
{
    TArray<FMeshData> MeshArray;
    
    if (!JsonObject.IsValid())
    {
        return MeshArray;
    }
    
    const TArray<TSharedPtr<FJsonValue>>* MeshDataArray;
    if (!JsonObject->TryGetArrayField(TEXT("mesh_data"), MeshDataArray))
    {
        return MeshArray;
    }
    
    for (const auto& Value : *MeshDataArray)
    {
        if (Value->Type == EJson::Object)
        {
            MeshArray.Add(ParseSingleMesh(Value->AsObject()));
        }
    }
    
    return MeshArray;
}

TArray<FOrganVolumeStats> FJsonParser::ParseOrganStats(TSharedPtr<FJsonObject> JsonObject)
{
    TArray<FOrganVolumeStats> StatsArray;
    
    if (!JsonObject.IsValid())
    {
        return StatsArray;
    }
    
    const TArray<TSharedPtr<FJsonValue>>* OrganStatsArray;
    if (!JsonObject->TryGetArrayField(TEXT("organ_stats"), OrganStatsArray))
    {
        return StatsArray;
    }
    
    for (const auto& Value : *OrganStatsArray)
    {
        if (Value->Type == EJson::Object)
        {
            StatsArray.Add(ParseSingleOrganStats(Value->AsObject()));
        }
    }
    
    return StatsArray;
}

TSharedPtr<FJsonObject> FJsonParser::StringToJson(const FString& JsonString)
{
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
    
    if (FJsonSerializer::Deserialize(Reader, JsonObject))
    {
        return JsonObject;
    }
    
    return nullptr;
}

FMeshData FJsonParser::ParseSingleMesh(TSharedPtr<FJsonObject> MeshObject)
{
    FMeshData MeshData;
    
    if (!MeshObject.IsValid())
    {
        return MeshData;
    }
    
    MeshData.OrganName = MeshObject->GetStringField(TEXT("organ_name"));
    
    // Parse vertices
    const TArray<TSharedPtr<FJsonValue>>* VerticesArray;
    if (MeshObject->TryGetArrayField(TEXT("vertices"), VerticesArray))
    {
        for (const auto& VertexValue : *VerticesArray)
        {
            const TArray<TSharedPtr<FJsonValue>>* VertexCoords;
            if (VertexValue->TryGetArray(VertexCoords) && VertexCoords->Num() == 3)
            {
                float X = (*VertexCoords)[0]->AsNumber();
                float Y = (*VertexCoords)[1]->AsNumber();
                float Z = (*VertexCoords)[2]->AsNumber();
                MeshData.Vertices.Add(FVector(X * 10.0f, Y * 10.0f, Z * 10.0f));
            }
        }
    }
    
    // Parse triangles
    const TArray<TSharedPtr<FJsonValue>>* TrianglesArray;
    if (MeshObject->TryGetArrayField(TEXT("triangles"), TrianglesArray))
    {
        for (const auto& TriangleValue : *TrianglesArray)
        {
            MeshData.Triangles.Add(TriangleValue->AsNumber());
        }
    }
    
    // Set color based on organ type
    if (MeshData.OrganName.Contains(TEXT("Liver")))
    {
        MeshData.Color = FLinearColor(0.6f, 0.2f, 0.2f, 0.8f);
    }
    else if (MeshData.OrganName.Contains(TEXT("Vessel")))
    {
        MeshData.Color = FLinearColor(0.2f, 0.2f, 0.8f, 0.8f);
    }
    else if (MeshData.OrganName.Contains(TEXT("Tumor")))
    {
        MeshData.Color = FLinearColor(0.8f, 0.8f, 0.2f, 0.9f);
    }
    else
    {
        MeshData.Color = FLinearColor(0.5f, 0.5f, 0.5f, 0.8f);
    }
    
    return MeshData;
}

FOrganVolumeStats FJsonParser::ParseSingleOrganStats(TSharedPtr<FJsonObject> StatsObject)
{
    FOrganVolumeStats Stats;
    
    if (!StatsObject.IsValid())
    {
        return Stats;
    }
    
    Stats.OrganName = StatsObject->GetStringField(TEXT("organ_name"));
    Stats.VolumeML = StatsObject->GetNumberField(TEXT("volume_ml"));
    Stats.VoxelCount = StatsObject->GetIntegerField(TEXT("voxel_count"));
    
    if (StatsObject->HasField(TEXT("num_components")))
    {
        Stats.NumComponents = StatsObject->GetIntegerField(TEXT("num_components"));
    }
    
    return Stats;
}