#pragma once
#include "Widgets/SCompoundWidget.h"
#include "Types/LiverAITypes.h"

// Forward declaration
class SMultiLineEditableTextBox;

class LIVERIMAGEAI_API SDiagnosticReportWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SDiagnosticReportWidget) {}
    SLATE_END_ARGS()
    
    void Construct(const FArguments& InArgs);
    
    void ShowReport(const FLiverAnalysisResult& Result);
    void ShowRawText(const FString& Text);
    void ClearReport();
    
private:
    TSharedPtr<SMultiLineEditableTextBox> ReportText;
    
    FString GenerateDiagnosticReport(const FLiverAnalysisResult& Result);
    FString GenerateOrganStatsReport(const TArray<FOrganVolumeStats>& OrganStats);
    FString GenerateClinicalAssessment(const TArray<FOrganVolumeStats>& OrganStats);
};