#include "UI/ProgressWidget.h"
#include "Types/LiverAITypes.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SBorder.h"
#include "Styling/CoreStyle.h"

void SProgressWidget::Construct(const FArguments& InArgs)
{
    CurrentProgress = 0.0f;
    CurrentStatus = TEXT("Ready");
    
    ChildSlot
    [
        SNew(SVerticalBox)
        + SVerticalBox::Slot()
        .AutoHeight()
        [
            SAssignNew(ProgressText, STextBlock)
            .Text(FText::FromString(TEXT("Progress: 0%")))
            .Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
            .ColorAndOpacity(FLinearColor::White)
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SAssignNew(StatusText, STextBlock)
            .Text(FText::FromString(TEXT("Ready")))
            .Font(FCoreStyle::GetDefaultFontStyle("Regular", 9))
        ]
    ];
}

void SProgressWidget::UpdateProgress(const FAnalysisProgress& Progress)
{
    SetProgress(Progress.Progress, Progress.Message);
}

void SProgressWidget::SetProgress(float ProgressValue, const FString& Message)
{
    CurrentProgress = FMath::Clamp(ProgressValue, 0.0f, 100.0f);
    CurrentStatus = Message;
    
    if (ProgressText.IsValid())
    {
        FString ProgressStr = FString::Printf(TEXT("Progress: %.1f%%"), CurrentProgress);
        ProgressText->SetText(FText::FromString(ProgressStr));
        
        // Update color based on progress
        FLinearColor ProgressColor = FLinearColor::White;
        if (CurrentProgress >= 100.0f)
        {
            ProgressColor = FLinearColor::Green;
        }
        else if (CurrentProgress > 0.0f)
        {
            ProgressColor = FLinearColor::Yellow;
        }
        ProgressText->SetColorAndOpacity(ProgressColor);
    }
    
    if (StatusText.IsValid())
    {
        StatusText->SetText(FText::FromString(CurrentStatus));
    }
}

void SProgressWidget::Reset()
{
    SetProgress(0.0f, TEXT("Ready"));
}