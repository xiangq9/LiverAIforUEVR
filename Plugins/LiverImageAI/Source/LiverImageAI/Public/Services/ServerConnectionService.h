#pragma once
#include "Interfaces/IServerConnectionService.h"

class FHttpService;

class LIVERIMAGEAI_API FServerConnectionService : public IServerConnectionService
{
public:
    FServerConnectionService();
    virtual ~FServerConnectionService();
    
    // IServerConnectionService interface implementation
    virtual void TestConnection(const FString& ServerURL) override;
    virtual bool IsServerHealthy() const override;
    virtual void SetServerURL(const FString& URL) override;
    virtual FString GetServerURL() const override;
    virtual FString GetServerStatus() const override;
    
private:
    TSharedPtr<FHttpService> HttpService;
    FString CurrentServerURL;
    bool bServerHealthy;
    FString ServerStatusMessage;
    
    // Internal callback
    void OnHealthCheckComplete(bool bSuccess, const FString& Response);
};