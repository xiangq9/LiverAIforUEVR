#pragma once
#include "Interfaces/IAnalysisService.h"
#include "Types/LiverAITypes.h"
#include "Types/ServerConfig.h"

class FHttpService;
class FJsonParser;
class FProgressMonitor;

class LIVERIMAGEAI_API FAnalysisService : public IAnalysisService
{
public:
    FAnalysisService();
    virtual ~FAnalysisService();
    
    // IAnalysisService interface implementation
    virtual void StartAnalysis(const FLiverAnalysisRequest& Request) override;
    virtual void CancelAnalysis() override;
    virtual void CheckProgress(const FString& RequestId) override;
    virtual void GetResult(const FString& RequestId) override;
    virtual bool IsAnalysisInProgress() const override;
    virtual FString GetCurrentRequestId() const override;
    
    // Configuration
    void SetServerConfig(const FServerConfig& Config);
    
private:
    TSharedPtr<FHttpService> HttpService;
    TSharedPtr<FJsonParser> JsonParser;
    TSharedPtr<FProgressMonitor> ProgressMonitor;
    
    FServerConfig ServerConfig;
    FString CurrentRequestId;
    bool bAnalysisInProgress;
    
    // Internal callback methods
    void OnAnalysisRequestComplete(bool bSuccess, const FString& Response);
    void OnProgressCheckComplete(bool bSuccess, const FString& Response);
    void OnResultReceived(bool bSuccess, const FString& Response);
    void OnProgressTimerTick();
};