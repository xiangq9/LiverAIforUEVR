#pragma once
#include "Widgets/SCompoundWidget.h"
#include "Types/LiverAITypes.h"

class LIVERIMAGEAI_API SProgressWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SProgressWidget) {}
    SLATE_END_ARGS()
    
    void Construct(const FArguments& InArgs);
    
    void UpdateProgress(const FAnalysisProgress& Progress);
    void SetProgress(float ProgressValue, const FString& Message);
    void Reset();
    
private:
    TSharedPtr<STextBlock> ProgressText;
    TSharedPtr<STextBlock> StatusText;
    
    float CurrentProgress;
    FString CurrentStatus;
};