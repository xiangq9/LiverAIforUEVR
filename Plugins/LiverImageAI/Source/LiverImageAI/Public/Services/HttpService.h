#pragma once
#include "Interfaces/IHttpService.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "HttpModule.h"

class LIVERIMAGEAI_API FHttpService : public IHttpService
{
public:
    FHttpService();
    virtual ~FHttpService();
    
    // IHttpService interface
    virtual void SendRequest(const FString& URL, const FString& Verb, const FString& Content) override;
    virtual void CancelPendingRequests() override;
    virtual bool IsRequestInProgress() const override;
    virtual void SetTimeout(float Timeout) override;
    
    FOnRequestComplete OnRequestComplete;
    FOnProgressUpdate OnProgressUpdate;
    FOnResultReceived OnResultReceived;
    
private:
    mutable TArray<TSharedPtr<IHttpRequest, ESPMode::ThreadSafe>> PendingRequests;
    float RequestTimeout;
    
    void OnHttpRequestComplete(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful);
};