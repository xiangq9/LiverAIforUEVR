#pragma once
#include "CoreMinimal.h"

class LIVERIMAGEAI_API IServerConnectionService
{
public:
    virtual ~IServerConnectionService() = default;
    
    // Delegate declaration
    DECLARE_MULTICAST_DELEGATE_TwoParams(FOnConnectionStatusChanged, bool, const FString&);
    
    // Delegate instance
    FOnConnectionStatusChanged OnConnectionStatusChanged;
    
    // Pure virtual methods
    virtual void TestConnection(const FString& ServerURL) = 0;
    virtual bool IsServerHealthy() const = 0;
    virtual void SetServerURL(const FString& URL) = 0;
    virtual FString GetServerURL() const = 0;
    virtual FString GetServerStatus() const = 0;
};