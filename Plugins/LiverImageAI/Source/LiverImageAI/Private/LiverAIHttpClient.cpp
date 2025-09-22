#include "LiverAIHttpClient.h"
#include "HttpModule.h"
#include "Json.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Misc/DateTime.h"
#include "HAL/PlatformFilemanager.h"
#include "Engine/Engine.h"

ULiverAIHttpClient::ULiverAIHttpClient()
{
    PythonServerURL = TEXT("http://127.0.0.1:8888");
    RequestTimeout = 300.0f; // 5 minute timeout
    bServerHealthy = false;
}

void ULiverAIHttpClient::SetServerURL(const FString& URL)
{
    PythonServerURL = URL;
    UE_LOG(LogTemp, Log, TEXT("Server URL set to: %s"), *URL);
}

bool ULiverAIHttpClient::IsServerHealthy() const
{
    return bServerHealthy;
}

FString ULiverAIHttpClient::GetServerURL() const
{
    return PythonServerURL;
}

void ULiverAIHttpClient::CheckServerHealth()
{
    if (!FHttpModule::Get().IsHttpEnabled())
    {
        UE_LOG(LogTemp, Error, TEXT("HTTP module not enabled"));
        OnServerHealthCheck.Broadcast(false, TEXT("HTTP module not enabled"));
        return;
    }

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    
    HttpRequest->OnProcessRequestComplete().BindUObject(this, &ULiverAIHttpClient::OnHealthCheckResponse);
    HttpRequest->SetVerb("GET");
    HttpRequest->SetURL(PythonServerURL + TEXT("/api/health"));
    HttpRequest->SetTimeout(10.0f);
    
    if (!HttpRequest->ProcessRequest())
    {
        UE_LOG(LogTemp, Error, TEXT("Unable to send health check request"));
        OnServerHealthCheck.Broadcast(false, TEXT("Unable to send request"));
    }
}

void ULiverAIHttpClient::SendAnalysisRequest(const FLiverAnalysisRequest& Request)
{
    if (!FHttpModule::Get().IsHttpEnabled())
    {
        UE_LOG(LogTemp, Error, TEXT("HTTP module not enabled"));
        return;
    }

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    
    HttpRequest->OnProcessRequestComplete().BindUObject(this, &ULiverAIHttpClient::OnAnalysisRequestResponse);
    HttpRequest->SetVerb("POST");
    HttpRequest->SetURL(PythonServerURL + TEXT("/api/analyze"));
    HttpRequest->SetHeader("Content-Type", TEXT("application/json"));
    HttpRequest->SetTimeout(RequestTimeout);
    
    // Build JSON payload
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
    JsonObject->SetStringField(TEXT("mri_file_path"), Request.MRIFilePath);
    JsonObject->SetStringField(TEXT("liver_model_path"), Request.LiverModelPath);
    JsonObject->SetStringField(TEXT("vessel_model_path"), Request.VesselModelPath);
    JsonObject->SetStringField(TEXT("tumor_model_path"), Request.TumorModelPath);
    JsonObject->SetStringField(TEXT("request_id"), Request.RequestId);
    
    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);
    
    HttpRequest->SetContentAsString(OutputString);
    
    if (!HttpRequest->ProcessRequest())
    {
        UE_LOG(LogTemp, Error, TEXT("Unable to send analysis request"));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Sending analysis request: %s"), *Request.RequestId);
    }
}

void ULiverAIHttpClient::GetAnalysisProgress(const FString& RequestId)
{
    if (!FHttpModule::Get().IsHttpEnabled() || RequestId.IsEmpty())
    {
        return;
    }

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    
    HttpRequest->OnProcessRequestComplete().BindUObject(this, &ULiverAIHttpClient::OnProgressResponse);
    HttpRequest->SetVerb("GET");
    HttpRequest->SetURL(PythonServerURL + TEXT("/api/status/") + RequestId);
    HttpRequest->SetTimeout(10.0f);
    
    HttpRequest->ProcessRequest();
}

void ULiverAIHttpClient::GetAnalysisResult(const FString& RequestId)
{
    if (!FHttpModule::Get().IsHttpEnabled() || RequestId.IsEmpty())
    {
        return;
    }

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    
    HttpRequest->OnProcessRequestComplete().BindUObject(this, &ULiverAIHttpClient::OnResultResponse);
    HttpRequest->SetVerb("GET");
    HttpRequest->SetURL(PythonServerURL + TEXT("/api/result/") + RequestId);
    HttpRequest->SetTimeout(30.0f);
    
    HttpRequest->ProcessRequest();
}

// HTTP response handler implementation
void ULiverAIHttpClient::OnHealthCheckResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully)
{
    bool bHealthy = false;
    FString StatusMessage;
    
    if (bConnectedSuccessfully && Response.IsValid())
    {
        int32 ResponseCode = Response->GetResponseCode();
        UE_LOG(LogTemp, Log, TEXT("Health check response code: %d"), ResponseCode);
        
        if (ResponseCode == 200)
        {
            TSharedPtr<FJsonObject> JsonObject = ParseJsonResponse(Response);
            if (JsonObject.IsValid())
            {
                FString Status = JsonObject->GetStringField(TEXT("status"));
                bool bAIModules = JsonObject->GetBoolField(TEXT("ai_modules"));
                FString Device = JsonObject->GetStringField(TEXT("device"));
                
                bHealthy = (Status == TEXT("healthy") && bAIModules);
                StatusMessage = FString::Printf(TEXT("Status: %s, AI Modules: %s, Device: %s"), 
                    *Status, bAIModules ? TEXT("Available") : TEXT("Unavailable"), *Device);
                
                UE_LOG(LogTemp, Log, TEXT("Server health check successful: %s"), *StatusMessage);
            }
            else
            {
                StatusMessage = TEXT("Invalid JSON response");
                UE_LOG(LogTemp, Warning, TEXT("Health check response JSON parsing failed"));
            }
        }
        else
        {
            StatusMessage = FString::Printf(TEXT("Server response error: %d"), ResponseCode);
            UE_LOG(LogTemp, Warning, TEXT("Health check failed, response code: %d"), ResponseCode);
        }
    }
    else
    {
        StatusMessage = TEXT("Unable to connect to Python backend server");
        UE_LOG(LogTemp, Warning, TEXT("Health check connection failed"));
    }
    
    bServerHealthy = bHealthy;
    OnServerHealthCheck.Broadcast(bHealthy, StatusMessage);
}

void ULiverAIHttpClient::OnAnalysisRequestResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully)
{
    if (!bConnectedSuccessfully || !Response.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Analysis request failed: No response"));
        
        // Broadcast failure result
        FLiverAnalysisResult FailResult;
        FailResult.bSuccess = false;
        FailResult.ErrorMessage = TEXT("Network connection failed");
        OnAnalysisComplete.Broadcast(FailResult);
        return;
    }
    
    int32 ResponseCode = Response->GetResponseCode();
    FString ResponseContent = Response->GetContentAsString();
    
    UE_LOG(LogTemp, Log, TEXT("Analysis request response: %d - %s"), ResponseCode, *ResponseContent);
    
    if (ResponseCode == 200)
    {
        TSharedPtr<FJsonObject> JsonObject = ParseJsonResponse(Response);
        if (JsonObject.IsValid())
        {
            FString RequestId = JsonObject->GetStringField(TEXT("request_id"));
            FString Status = JsonObject->GetStringField(TEXT("status"));
            UE_LOG(LogTemp, Log, TEXT("Analysis started successfully: %s, Status: %s"), *RequestId, *Status);
        }
    }
    else
    {
        // Broadcast failure result
        FLiverAnalysisResult FailResult;
        FailResult.bSuccess = false;
        FailResult.ErrorMessage = FString::Printf(TEXT("Server error: %d"), ResponseCode);
        OnAnalysisComplete.Broadcast(FailResult);
    }
}

void ULiverAIHttpClient::OnProgressResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully)
{
    if (bConnectedSuccessfully && Response.IsValid() && Response->GetResponseCode() == 200)
    {
        TSharedPtr<FJsonObject> JsonObject = ParseJsonResponse(Response);
        if (JsonObject.IsValid())
        {
            FAnalysisProgress Progress = ParseAnalysisProgress(JsonObject);
            OnAnalysisProgress.Broadcast(Progress);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to get progress"));
    }
}

void ULiverAIHttpClient::OnResultResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully)
{
    FLiverAnalysisResult Result;
    
    if (bConnectedSuccessfully && Response.IsValid())
    {
        int32 ResponseCode = Response->GetResponseCode();
        if (ResponseCode == 200)
        {
            TSharedPtr<FJsonObject> JsonObject = ParseJsonResponse(Response);
            if (JsonObject.IsValid())
            {
                Result = ParseAnalysisResult(JsonObject);
                UE_LOG(LogTemp, Log, TEXT("Analysis result retrieved successfully: %s"), *Result.RequestId);
            }
            else
            {
                Result.bSuccess = false;
                Result.ErrorMessage = TEXT("Result JSON parsing failed");
            }
        }
        else if (ResponseCode == 404)
        {
            Result.bSuccess = false;
            Result.ErrorMessage = TEXT("Analysis result not found");
        }
        else
        {
            Result.bSuccess = false;
            Result.ErrorMessage = FString::Printf(TEXT("Server error: %d"), ResponseCode);
        }
    }
    else
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Connection failed");
    }
    
    OnAnalysisComplete.Broadcast(Result);
}

// JSON parsing helper function implementation
TSharedPtr<FJsonObject> ULiverAIHttpClient::ParseJsonResponse(FHttpResponsePtr Response)
{
    if (!Response.IsValid())
        return nullptr;
        
    FString ResponseContent = Response->GetContentAsString();
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseContent);
    
    if (FJsonSerializer::Deserialize(Reader, JsonObject))
    {
        return JsonObject;
    }
    
    UE_LOG(LogTemp, Warning, TEXT("JSON parsing failed: %s"), *ResponseContent);
    return nullptr;
}

FAnalysisProgress ULiverAIHttpClient::ParseAnalysisProgress(TSharedPtr<FJsonObject> JsonObject)
{
    FAnalysisProgress Progress;
    
    if (JsonObject.IsValid())
    {
        Progress.Status = JsonObject->GetStringField(TEXT("status"));
        Progress.Progress = JsonObject->GetNumberField(TEXT("progress"));
        Progress.Message = JsonObject->GetStringField(TEXT("message"));
        
        // If no message field, use status
        if (Progress.Message.IsEmpty())
        {
            Progress.Message = Progress.Status;
        }
    }
    
    return Progress;
}

FLiverAnalysisResult ULiverAIHttpClient::ParseAnalysisResult(TSharedPtr<FJsonObject> JsonObject)
{
    FLiverAnalysisResult Result;
    
    if (!JsonObject.IsValid())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Invalid JSON data");
        return Result;
    }
    
    Result.bSuccess = JsonObject->GetBoolField(TEXT("success"));
    Result.RequestId = JsonObject->GetStringField(TEXT("request_id"));
    Result.ErrorMessage = JsonObject->GetStringField(TEXT("error_message"));
    Result.ResultSummary = JsonObject->GetStringField(TEXT("diagnostic_report"));
    
    // Parse organ statistics data
    const TArray<TSharedPtr<FJsonValue>>* OrganStatsArray;
    if (JsonObject->TryGetArrayField(TEXT("organ_stats"), OrganStatsArray))
    {
        Result.OrganStats = ParseOrganStats(*OrganStatsArray);
    }
    
    return Result;
}

TArray<FOrganVolumeStats> ULiverAIHttpClient::ParseOrganStats(const TArray<TSharedPtr<FJsonValue>>& JsonArray)
{
    TArray<FOrganVolumeStats> OrganStats;
    
    for (const auto& JsonValue : JsonArray)
    {
        if (JsonValue->Type == EJson::Object)
        {
            TSharedPtr<FJsonObject> StatsObject = JsonValue->AsObject();
            FOrganVolumeStats Stats;
            
            Stats.OrganName = StatsObject->GetStringField(TEXT("organ_name"));
            Stats.VolumeML = StatsObject->GetNumberField(TEXT("volume_ml"));
            Stats.VoxelCount = StatsObject->GetIntegerField(TEXT("voxel_count"));
            
            OrganStats.Add(Stats);
        }
    }
    
    return OrganStats;
}

// Utility function complete implementation
FString ULiverAIUtilities::GenerateRequestId()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("ue_request_%s"), *Now.ToString(TEXT("%Y%m%d_%H%M%S")));
}

FLinearColor ULiverAIUtilities::GetOrganColor(const FString& OrganName)
{
    FString LowerName = OrganName.ToLower();
    
    if (LowerName.Contains(TEXT("liver")))
        return FLinearColor(0.0f, 0.8f, 0.0f, 0.8f); // Green
    else if (LowerName.Contains(TEXT("vessel")))
        return FLinearColor(0.8f, 0.0f, 0.0f, 0.8f); // Red
    else if (LowerName.Contains(TEXT("tumor")))
        return FLinearColor(0.8f, 0.8f, 0.0f, 0.9f); // Yellow
    else
        return FLinearColor(0.7f, 0.7f, 0.7f, 0.8f); // Gray
}

FString ULiverAIUtilities::GetCurrentTimeString()
{
    FDateTime Now = FDateTime::Now();
    return Now.ToString(TEXT("%Y-%m-%d %H:%M:%S"));
}

bool ULiverAIUtilities::ValidateFilePath(const FString& FilePath)
{
    return FPlatformFileManager::Get().GetPlatformFile().FileExists(*FilePath);
}

FString ULiverAIUtilities::FormatFileSize(int64 SizeInBytes)
{
    if (SizeInBytes < 1024)
        return FString::Printf(TEXT("%lld B"), SizeInBytes);
    else if (SizeInBytes < 1024 * 1024)
        return FString::Printf(TEXT("%.1f KB"), SizeInBytes / 1024.0);
    else if (SizeInBytes < 1024 * 1024 * 1024)
        return FString::Printf(TEXT("%.1f MB"), SizeInBytes / (1024.0 * 1024.0));
    else
        return FString::Printf(TEXT("%.1f GB"), SizeInBytes / (1024.0 * 1024.0 * 1024.0));
}

FString ULiverAIUtilities::GetFileExtension(const FString& FilePath)
{
    return FPaths::GetExtension(FilePath);
}
