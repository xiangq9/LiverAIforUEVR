#pragma once
#include "CoreMinimal.h"

class LIVERIMAGEAI_API IProgressMonitor
{
public:
    virtual ~IProgressMonitor() = default;
    
    DECLARE_MULTICAST_DELEGATE(FOnProgressTick);
    
    virtual void StartMonitoring(float Interval = 3.0f) = 0;
    virtual void StopMonitoring() = 0;
    virtual bool IsMonitoring() const = 0;
    virtual void SetCheckInterval(float Interval) = 0;
    virtual float GetElapsedTime() const = 0;
};