#include "Services/HttpService.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Json.h"
#include "Types/LiverAITypes.h"

FHttpService::FHttpService()
    : RequestTimeout(300.0f)
{
}

FHttpService::~FHttpService()
{
    CancelPendingRequests();
}

void FHttpService::SendRequest(const FString& URL, const FString& Verb, const FString& Content)
{
    if (!FHttpModule::Get().IsHttpEnabled())
    {
        OnRequestComplete.Broadcast(false, TEXT("HTTP module not enabled"));
        return;
    }

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    
    HttpRequest->OnProcessRequestComplete().BindRaw(this, &FHttpService::OnHttpRequestComplete);
    HttpRequest->SetVerb(Verb);
    HttpRequest->SetURL(URL);
    HttpRequest->SetTimeout(RequestTimeout);
    
    if (!Content.IsEmpty())
    {
        HttpRequest->SetHeader("Content-Type", TEXT("application/json"));
        HttpRequest->SetContentAsString(Content);
    }
    
    PendingRequests.Add(HttpRequest);
    
    if (!HttpRequest->ProcessRequest())
    {
        OnRequestComplete.Broadcast(false, TEXT("Failed to send request"));
        PendingRequests.Remove(HttpRequest);
    }
}

void FHttpService::CancelPendingRequests()
{
    for (auto& Request : PendingRequests)
    {
        if (Request.IsValid())
        {
            Request->CancelRequest();
        }
    }
    PendingRequests.Empty();
}

bool FHttpService::IsRequestInProgress() const
{
    // Remove completed requests
    PendingRequests.RemoveAll([](const TSharedPtr<IHttpRequest, ESPMode::ThreadSafe>& Request)
    {
        return !Request.IsValid() || 
               Request->GetStatus() == EHttpRequestStatus::Failed || 
               Request->GetStatus() == EHttpRequestStatus::Succeeded;
    });
    
    return PendingRequests.Num() > 0;
}

void FHttpService::SetTimeout(float Timeout)
{
    RequestTimeout = FMath::Max(1.0f, Timeout);
}

void FHttpService::OnHttpRequestComplete(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful)
{
    if (bWasSuccessful && Response.IsValid())
    {
        int32 ResponseCode = Response->GetResponseCode();
        FString ResponseContent = Response->GetContentAsString();
        
        if (ResponseCode == 200)
        {
            OnRequestComplete.Broadcast(true, ResponseContent);
        }
        else
        {
            OnRequestComplete.Broadcast(false, FString::Printf(TEXT("HTTP Error: %d"), ResponseCode));
        }
    }
    else
    {
        OnRequestComplete.Broadcast(false, TEXT("Request failed"));
    }
    
    PendingRequests.Remove(Request);
}