#include "Services/AnalysisService.h"
#include "Services/HttpService.h"
#include "Utils/JsonParser.h"
#include "Utils/ProgressMonitor.h"
#include "Json.h"

FAnalysisService::FAnalysisService()
    : bAnalysisInProgress(false)
{
    HttpService = MakeShareable(new FHttpService());
    JsonParser = MakeShareable(new FJsonParser());
    ProgressMonitor = MakeShareable(new FProgressMonitor());
    
    ProgressMonitor->OnProgressTick.AddRaw(this, &FAnalysisService::OnProgressTimerTick);
}

FAnalysisService::~FAnalysisService()
{
    CancelAnalysis();
}

void FAnalysisService::StartAnalysis(const FLiverAnalysisRequest& Request)
{
    if (bAnalysisInProgress)
    {
        OnAnalysisError.Broadcast(TEXT("Analysis already in progress"));
        return;
    }
    
    CurrentRequestId = Request.RequestId;
    bAnalysisInProgress = true;
    
    // Build JSON request
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
    JsonObject->SetStringField(TEXT("mri_file_path"), Request.MRIFilePath);
    JsonObject->SetStringField(TEXT("liver_model_path"), Request.LiverModelPath);
    JsonObject->SetStringField(TEXT("vessel_model_path"), Request.VesselModelPath);
    JsonObject->SetStringField(TEXT("tumor_model_path"), Request.TumorModelPath);
    JsonObject->SetStringField(TEXT("request_id"), Request.RequestId);
    JsonObject->SetBoolField(TEXT("generate_3d_mesh"), Request.bGenerate3DMesh);
    
    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);
    
    // Send request
    FString URL = ServerConfig.ServerURL + TEXT("/api/analyze");
    HttpService->OnRequestComplete.AddRaw(this, &FAnalysisService::OnAnalysisRequestComplete);
    HttpService->SendRequest(URL, TEXT("POST"), OutputString);
    
    OnAnalysisStarted.Broadcast(CurrentRequestId);
}

void FAnalysisService::CancelAnalysis()
{
    if (bAnalysisInProgress)
    {
        bAnalysisInProgress = false;
        CurrentRequestId = TEXT("");
        HttpService->CancelPendingRequests();
        ProgressMonitor->StopMonitoring();
    }
}

void FAnalysisService::CheckProgress(const FString& RequestId)
{
    FString URL = ServerConfig.ServerURL + TEXT("/api/status/") + RequestId;
    HttpService->OnRequestComplete.AddRaw(this, &FAnalysisService::OnProgressCheckComplete);
    HttpService->SendRequest(URL, TEXT("GET"), TEXT(""));
}

void FAnalysisService::GetResult(const FString& RequestId)
{
    FString URL = ServerConfig.ServerURL + TEXT("/api/result/") + RequestId;
    HttpService->OnRequestComplete.AddRaw(this, &FAnalysisService::OnResultReceived);
    HttpService->SendRequest(URL, TEXT("GET"), TEXT(""));
}

bool FAnalysisService::IsAnalysisInProgress() const
{
    return bAnalysisInProgress;
}

FString FAnalysisService::GetCurrentRequestId() const
{
    return CurrentRequestId;
}

void FAnalysisService::SetServerConfig(const FServerConfig& Config)
{
    ServerConfig = Config;
    HttpService->SetTimeout(Config.RequestTimeout);
}

void FAnalysisService::OnAnalysisRequestComplete(bool bSuccess, const FString& Response)
{
    if (!bSuccess)
    {
        bAnalysisInProgress = false;
        OnAnalysisError.Broadcast(Response);
        return;
    }
    
    // Start progress monitoring
    ProgressMonitor->StartMonitoring(ServerConfig.ProgressCheckInterval);
}

void FAnalysisService::OnProgressCheckComplete(bool bSuccess, const FString& Response)
{
    if (!bSuccess || !bAnalysisInProgress)
    {
        return;
    }
    
    FAnalysisProgress Progress = JsonParser->ParseProgressUpdate(Response);
    OnAnalysisProgress.Broadcast(Progress);
    
    if (Progress.Status == TEXT("completed") || Progress.Progress >= 100.0f)
    {
        ProgressMonitor->StopMonitoring();
        GetResult(CurrentRequestId);
    }
    else if (Progress.Status == TEXT("failed"))
    {
        bAnalysisInProgress = false;
        ProgressMonitor->StopMonitoring();
        OnAnalysisError.Broadcast(Progress.Message);
    }
}

void FAnalysisService::OnResultReceived(bool bSuccess, const FString& Response)
{
    bAnalysisInProgress = false;
    ProgressMonitor->StopMonitoring();
    
    if (!bSuccess)
    {
        OnAnalysisError.Broadcast(Response);
        return;
    }
    
    FLiverAnalysisResult Result = JsonParser->ParseAnalysisResult(Response);
    OnAnalysisCompleted.Broadcast(Result);
}

void FAnalysisService::OnProgressTimerTick()
{
    if (bAnalysisInProgress && !CurrentRequestId.IsEmpty())
    {
        CheckProgress(CurrentRequestId);
    }
}