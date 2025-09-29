#include "Utils/ProgressMonitor.h"
#include "Engine/World.h"
#include "Editor.h"
#include "TimerManager.h"

FProgressMonitor::FProgressMonitor()
    : bIsMonitoring(false)
    , CheckInterval(3.0f)
    , ElapsedTime(0.0f)
{
}

FProgressMonitor::~FProgressMonitor()
{
    StopMonitoring();
}

void FProgressMonitor::StartMonitoring(float Interval)
{
    if (bIsMonitoring)
    {
        return;
    }
    
    CheckInterval = FMath::Max(0.1f, Interval);
    ElapsedTime = 0.0f;
    bIsMonitoring = true;
    
    if (GEditor)
    {
        GEditor->GetTimerManager()->SetTimer(
            TimerHandle,
            FTimerDelegate::CreateRaw(this, &FProgressMonitor::OnTimerTick),
            CheckInterval,
            true
        );
    }
}

void FProgressMonitor::StopMonitoring()
{
    if (!bIsMonitoring)
    {
        return;
    }
    
    bIsMonitoring = false;
    
    if (GEditor && TimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(TimerHandle);
    }
}

bool FProgressMonitor::IsMonitoring() const
{
    return bIsMonitoring;
}

void FProgressMonitor::SetCheckInterval(float Interval)
{
    CheckInterval = FMath::Max(0.1f, Interval);
    
    if (bIsMonitoring)
    {
        StopMonitoring();
        StartMonitoring(CheckInterval);
    }
}

float FProgressMonitor::GetElapsedTime() const
{
    return ElapsedTime;
}

void FProgressMonitor::OnTimerTick()
{
    ElapsedTime += CheckInterval;
    OnProgressTick.Broadcast();
}