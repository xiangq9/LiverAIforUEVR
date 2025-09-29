#include "UI/DiagnosticReportWidget.h"
#include "Widgets/Input/SMultiLineEditableTextBox.h"  // 正确的头文件
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"  // 正确路径
#include "Widgets/Text/STextBlock.h"
#include "Styling/CoreStyle.h"

void SDiagnosticReportWidget::Construct(const FArguments& InArgs)
{
    ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.GroupBorder"))
        .Padding(5)
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("AI Diagnostic Report")))
                .Font(FCoreStyle::GetDefaultFontStyle("Bold", 11))
            ]
            + SVerticalBox::Slot()
            .FillHeight(1.0f)
            .Padding(5)
            [
                SAssignNew(ReportText, SMultiLineEditableTextBox)
                .IsReadOnly(true)
                .Text(FText::FromString(TEXT("No analysis results yet.")))
                .Font(FCoreStyle::GetDefaultFontStyle("Mono", 8))
                .AlwaysShowScrollbars(true)
                .AutoWrapText(true)
            ]
        ]
    ];
}

void SDiagnosticReportWidget::ShowReport(const FLiverAnalysisResult& Result)
{
    FString Report = GenerateDiagnosticReport(Result);
    if (ReportText.IsValid())
    {
        ReportText->SetText(FText::FromString(Report));
    }
}

void SDiagnosticReportWidget::ShowRawText(const FString& Text)
{
    if (ReportText.IsValid())
    {
        ReportText->SetText(FText::FromString(Text));
    }
}

void SDiagnosticReportWidget::ClearReport()
{
    if (ReportText.IsValid())
    {
        ReportText->SetText(FText::FromString(TEXT("No analysis results yet.")));
    }
}

FString SDiagnosticReportWidget::GenerateDiagnosticReport(const FLiverAnalysisResult& Result)
{
    FString Report;
    Report += TEXT("=====================================\n");
    Report += TEXT("   AI Liver Analysis - Diagnostic Report\n");
    Report += TEXT("=====================================\n\n");
    Report += FString::Printf(TEXT("Analysis Time: %s\n"), *Result.Timestamp);
    Report += FString::Printf(TEXT("Request ID: %s\n\n"), *Result.RequestId);
    
    if (!Result.bSuccess)
    {
        Report += TEXT("【Analysis Failed】\n");
        Report += FString::Printf(TEXT("Error: %s\n"), *Result.ErrorMessage);
        return Report;
    }
    
    if (Result.OrganStats.Num() > 0)
    {
        Report += GenerateOrganStatsReport(Result.OrganStats);
        Report += GenerateClinicalAssessment(Result.OrganStats);
    }
    
    if (!Result.ResultSummary.IsEmpty())
    {
        Report += TEXT("\n【Additional Information】\n");
        Report += TEXT("─────────────────────────────\n");
        Report += Result.ResultSummary + TEXT("\n");
    }
    
    Report += TEXT("\n【Technical Notes】\n");
    Report += TEXT("─────────────────────────────\n");
    Report += TEXT("• Analysis performed using deep learning AI models\n");
    Report += TEXT("• 3D reconstruction using Marching Cubes algorithm\n");
    Report += TEXT("• Results for research/educational purposes only\n");
    Report += TEXT("=====================================\n");
    
    return Report;
}

FString SDiagnosticReportWidget::GenerateOrganStatsReport(const TArray<FOrganVolumeStats>& OrganStats)
{
    FString Report = TEXT("【Organ Segmentation Results】\n");
    Report += TEXT("─────────────────────────────\n\n");
    
    for (const FOrganVolumeStats& Stats : OrganStats)
    {
        Report += FString::Printf(TEXT("● %s:\n"), *Stats.OrganName);
        Report += FString::Printf(TEXT("  Volume: %.2f mL\n"), Stats.VolumeML);
        Report += FString::Printf(TEXT("  Voxel Count: %d\n"), Stats.VoxelCount);
        if (Stats.NumComponents > 1)
        {
            Report += FString::Printf(TEXT("  Connected Components: %d\n"), Stats.NumComponents);
        }
        Report += TEXT("\n");
    }
    
    return Report;
}

FString SDiagnosticReportWidget::GenerateClinicalAssessment(const TArray<FOrganVolumeStats>& OrganStats)
{
    FString Report = TEXT("\n【Clinical Assessment】\n");
    Report += TEXT("─────────────────────────────\n\n");
    
    float TotalLiverVolume = 0.0f;
    float TotalTumorVolume = 0.0f;
    int32 TumorCount = 0;
    
    for (const FOrganVolumeStats& Stats : OrganStats)
    {
        if (Stats.OrganName.Contains(TEXT("liver")))
        {
            TotalLiverVolume = Stats.VolumeML;
        }
        else if (Stats.OrganName.Contains(TEXT("tumor")))
        {
            TotalTumorVolume = Stats.VolumeML;
            TumorCount = Stats.NumComponents;
        }
    }
    
    // Liver volume assessment
    if (TotalLiverVolume > 0)
    {
        if (TotalLiverVolume < 800)
        {
            Report += TEXT("⚠ Liver volume is small - Consider possible liver atrophy\n");
        }
        else if (TotalLiverVolume > 2200)
        {
            Report += TEXT("⚠ Liver volume is enlarged - Consider possible hepatomegaly\n");
        }
        else
        {
            Report += TEXT("✓ Liver volume within normal range\n");
        }
    }
    
    // Tumor assessment
    if (TumorCount > 0)
    {
        if (TumorCount == 1)
        {
            Report += FString::Printf(TEXT("⚠ Single lesion detected (%.1f mL)\n"), TotalTumorVolume);
        }
        else
        {
            Report += FString::Printf(TEXT("⚠ Multiple lesions detected (%d lesions, %.1f mL)\n"), 
                TumorCount, TotalTumorVolume);
        }
    }
    else
    {
        Report += TEXT("✓ No significant hepatic lesions detected\n");
    }
    
    return Report;
}