#pragma once
#include "CoreMinimal.h"

struct LIVERIMAGEAI_API FServerConfig
{
    FString ServerURL;
    float RequestTimeout;
    float ProgressCheckInterval;
    int32 MaxRetries;
    
    FServerConfig()
    {
        ServerURL = TEXT("http://127.0.0.1:8888");
        RequestTimeout = 300.0f;
        ProgressCheckInterval = 3.0f;
        MaxRetries = 3;
    }
};