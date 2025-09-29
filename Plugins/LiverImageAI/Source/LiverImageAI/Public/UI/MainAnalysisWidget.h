#pragma once
#include "Widgets/SCompoundWidget.h"
#include "Types/LiverAITypes.h"

// Forward declarations
class IAnalysisService;
class IServerConnectionService;
class IVisualizationService;
class SFileSelectionWidget;
class SProgressWidget;
class SDiagnosticReportWidget;
class SSliceViewerWidget;
class SButton;
class SEditableTextBox;
class SCheckBox;
class STextBlock;

class LIVERIMAGEAI_API SMainAnalysisWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SMainAnalysisWidget) {}
    SLATE_END_ARGS()
    
    void Construct(const FArguments& InArgs);
    
private:
    // Services
    TSharedPtr<IAnalysisService> AnalysisService;
    TSharedPtr<IServerConnectionService> ConnectionService;
    TSharedPtr<IVisualizationService> VisualizationService;
    
    // UI Components
    TSharedPtr<SEditableTextBox> ServerURLText;
    TSharedPtr<STextBlock> ServerStatusText;
    TSharedPtr<SFileSelectionWidget> MRIFileSelector;
    TSharedPtr<SFileSelectionWidget> LiverModelSelector;
    TSharedPtr<SFileSelectionWidget> VesselModelSelector;
    TSharedPtr<SFileSelectionWidget> TumorModelSelector;
    TSharedPtr<SCheckBox> Generate3DMeshCheckBox;
    TSharedPtr<SProgressWidget> ProgressWidget;
    TSharedPtr<SDiagnosticReportWidget> ReportWidget;
    TSharedPtr<SSliceViewerWidget> OriginalSliceViewer;
    TSharedPtr<SSliceViewerWidget> SegmentationSliceViewer;
    TSharedPtr<STextBlock> StatusText;
    
    // Control buttons
    TSharedPtr<SButton> RunAnalysisButton;
    TSharedPtr<SButton> CancelAnalysisButton;
    TSharedPtr<SButton> TestConnectionButton;
    
    // View plane buttons
    TSharedPtr<SButton> AxialButton;
    TSharedPtr<SButton> CoronalButton;
    TSharedPtr<SButton> SagittalButton;
    
    // State
    bool bAnalysisInProgress;
    bool bGenerate3DMesh;
    bool bSliceSyncEnabled;
    FString CurrentRequestId;
    
    // UI Creation methods
    TSharedRef<SWidget> CreateServerConnectionSection();
    TSharedRef<SWidget> CreateFileSelectionSection();
    TSharedRef<SWidget> CreateSliceViewerSection();
    TSharedRef<SWidget> CreateControlButtonsSection();
    TSharedRef<SWidget> CreateDiagnosticReportSection();
    
    // Event handlers
    FReply OnTestServerConnectionClicked();
    FReply OnRunAnalysisClicked();
    FReply OnCancelAnalysisClicked();
    FReply OnLoadTestResultClicked();
    FReply OnToggleSyncClicked();
    void OnGenerate3DMeshStateChanged(ECheckBoxState NewState);
    
    // Service callbacks
    void OnServerConnectionChanged(bool bHealthy, const FString& Status);
    void OnAnalysisStarted(const FString& RequestId);
    void OnAnalysisProgress(const FAnalysisProgress& Progress);
    void OnAnalysisCompleted(const FLiverAnalysisResult& Result);
    void OnAnalysisError(const FString& Error);
    
    // Slice viewer methods
    void ChangeViewPlane(int32 Plane);
    void SetupSliceViewerSync();
    void OnOriginalSliceChanged(int32 NewSlice);
    void OnSegmentationSliceChanged(int32 NewSlice);
    
    // UI State methods
    void UpdateUIState();
    void UpdateStatusText(const FString& Message);
    bool ValidateInputFiles();
    FString GenerateRequestId();
    
    // Result processing
    void ProcessAnalysisResult(const FLiverAnalysisResult& Result);
    void LoadTestResult();
};