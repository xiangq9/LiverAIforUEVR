#pragma once
#include "Interfaces/IProgressMonitor.h"

class LIVERIMAGEAI_API FProgressMonitor : public IProgressMonitor
{
public:
    FProgressMonitor();
    virtual ~FProgressMonitor();
    
    // IProgressMonitor interface
    virtual void StartMonitoring(float Interval = 3.0f) override;
    virtual void StopMonitoring() override;
    virtual bool IsMonitoring() const override;
    virtual void SetCheckInterval(float Interval) override;
    virtual float GetElapsedTime() const override;
    
    FOnProgressTick OnProgressTick;
    
private:
    FTimerHandle TimerHandle;
    bool bIsMonitoring;
    float CheckInterval;
    float ElapsedTime;
    
    void OnTimerTick();
};