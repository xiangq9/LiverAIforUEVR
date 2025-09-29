#pragma once
#include "CoreMinimal.h"
#include "Types/LiverAITypes.h"

class LIVERIMAGEAI_API IAnalysisService
{
public:
    virtual ~IAnalysisService() = default;
    
    // Delegate declarations
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnAnalysisStarted, const FString&);
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnAnalysisCompleted, const FLiverAnalysisResult&);
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnAnalysisProgress, const FAnalysisProgress&);
    DECLARE_MULTICAST_DELEGATE_OneParam(FOnAnalysisError, const FString&);
    
    // Delegate instances
    FOnAnalysisStarted OnAnalysisStarted;
    FOnAnalysisCompleted OnAnalysisCompleted;
    FOnAnalysisProgress OnAnalysisProgress;
    FOnAnalysisError OnAnalysisError;
    
    // Pure virtual methods
    virtual void StartAnalysis(const FLiverAnalysisRequest& Request) = 0;
    virtual void CancelAnalysis() = 0;
    virtual void CheckProgress(const FString& RequestId) = 0;
    virtual void GetResult(const FString& RequestId) = 0;
    virtual bool IsAnalysisInProgress() const = 0;
    virtual FString GetCurrentRequestId() const = 0;
};