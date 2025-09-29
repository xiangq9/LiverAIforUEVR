#pragma once
#include "CoreMinimal.h"
#include "Types/LiverAITypes.h"

class LIVERIMAGEAI_API IHttpService
{
public:
    virtual ~IHttpService() = default;
    
    DECLARE_MULTICAST_DELEGATE_TwoParams(FOnRequestComplete, bool, const FString&);
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnProgressUpdate, const FAnalysisProgress&);
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnResultReceived, const FLiverAnalysisResult&);
    
    virtual void SendRequest(const FString& URL, const FString& Verb, const FString& Content) = 0;
    virtual void CancelPendingRequests() = 0;
    virtual bool IsRequestInProgress() const = 0;
    virtual void SetTimeout(float Timeout) = 0;
};