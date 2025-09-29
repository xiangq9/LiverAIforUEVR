#include "Services/ServerConnectionService.h"
#include "Services/HttpService.h"
#include "Json.h"

FServerConnectionService::FServerConnectionService()
    : bServerHealthy(false)
    , ServerStatusMessage(TEXT("Not tested"))
{
    HttpService = MakeShareable(new FHttpService());
    CurrentServerURL = TEXT("http://127.0.0.1:8888");
}

FServerConnectionService::~FServerConnectionService()
{
}

void FServerConnectionService::TestConnection(const FString& ServerURL)
{
    CurrentServerURL = ServerURL;
    FString URL = ServerURL + TEXT("/api/health");
    
    HttpService->OnRequestComplete.AddRaw(this, &FServerConnectionService::OnHealthCheckComplete);
    HttpService->SetTimeout(10.0f);
    HttpService->SendRequest(URL, TEXT("GET"), TEXT(""));
}

bool FServerConnectionService::IsServerHealthy() const
{
    return bServerHealthy;
}

void FServerConnectionService::SetServerURL(const FString& URL)
{
    CurrentServerURL = URL;
}

FString FServerConnectionService::GetServerURL() const
{
    return CurrentServerURL;
}

FString FServerConnectionService::GetServerStatus() const
{
    return ServerStatusMessage;
}

void FServerConnectionService::OnHealthCheckComplete(bool bSuccess, const FString& Response)
{
    if (!bSuccess)
    {
        bServerHealthy = false;
        ServerStatusMessage = TEXT("Connection failed");
        OnConnectionStatusChanged.Broadcast(false, ServerStatusMessage);
        return;
    }
    
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Response);
    
    if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
    {
        FString Status = JsonObject->GetStringField(TEXT("status"));
        bool bAIModules = JsonObject->GetBoolField(TEXT("ai_modules"));
        FString Device = JsonObject->GetStringField(TEXT("device"));
        
        bServerHealthy = (Status == TEXT("healthy") && bAIModules);
        ServerStatusMessage = FString::Printf(TEXT("Status: %s, AI: %s, Device: %s"), 
            *Status, bAIModules ? TEXT("Ready") : TEXT("Not Ready"), *Device);
    }
    else
    {
        bServerHealthy = false;
        ServerStatusMessage = TEXT("Invalid response");
    }
    
    OnConnectionStatusChanged.Broadcast(bServerHealthy, ServerStatusMessage);
}