#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Framework/Commands/Commands.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SMultiLineEditableTextBox.h"
#include "SLiverSliceViewer.h"
#include "LiverAITypes.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Framework/Application/SlateApplication.h"

// Forward declarations
class FToolBarBuilder;
class FMenuBuilder;
class FJsonObject;

// Plugin commands definition
class FLiverImageAICommands : public TCommands<FLiverImageAICommands>
{
public:
    FLiverImageAICommands()
        : TCommands<FLiverImageAICommands>(
            TEXT("LiverImageAI"),
            NSLOCTEXT("Contexts", "LiverImageAI", "LiverImageAI Plugin"),
            NAME_None,
            FAppStyle::GetAppStyleSetName())
    {
    }

    virtual void RegisterCommands() override;

public:
    TSharedPtr<FUICommandInfo> OpenPluginWindow;
};

// Main plugin module class
class FLiverImageAIModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
    void PluginButtonClicked();

private:
    void RegisterMenus();
    TSharedRef<class SDockTab> OnSpawnPluginTab(const class FSpawnTabArgs& SpawnTabArgs);
    TSharedPtr<class FUICommandList> PluginCommands;
};

// Main AI analysis widget
class SLiverAIWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SLiverAIWidget) {}
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

private:
    // UI Components
    TSharedPtr<SEditableTextBox> ServerURLText;
    TSharedPtr<SEditableTextBox> CTFilePathText;
    TSharedPtr<SEditableTextBox> LiverModelPathText;
    TSharedPtr<SEditableTextBox> VesselModelPathText;
    TSharedPtr<SEditableTextBox> TumorModelPathText;
    TSharedPtr<SCheckBox> Generate3DMeshCheckBox;
    TSharedPtr<SButton> RunAnalysisButton;
    TSharedPtr<SButton> CancelAnalysisButton;
    TSharedPtr<SButton> AxialButton;
    TSharedPtr<SButton> CoronalButton;
    TSharedPtr<SButton> SagittalButton;
    TSharedPtr<STextBlock> ServerStatusText;
    TSharedPtr<STextBlock> ProgressText;
    TSharedPtr<STextBlock> StatusText;
    TSharedPtr<SMultiLineEditableTextBox> DiagnosticResultsText;
    TSharedPtr<SLiverSliceViewer> OriginalSliceViewer;
    TSharedPtr<SLiverSliceViewer> SegmentationSliceViewer;

    // State Variables
    bool bServerHealthy;
    bool bAnalysisInProgress;
    bool bGenerate3DMesh;
    FString ServerURL;
    FString CurrentRequestId;
    float CurrentProgress;
    float LastProgressCheckTime;
    TWeakPtr<FActiveTimerHandle> ActiveTimerHandle;

    // File Operations
    bool OpenFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename);

    // UI Event Handlers
    FReply OnTestServerConnectionClicked();
    FReply OnBrowseCTFileClicked();
    FReply OnBrowseLiverModelClicked();
    FReply OnBrowseVesselModelClicked();
    FReply OnBrowseTumorModelClicked();
    FReply OnRunAnalysisClicked();
    FReply OnCancelAnalysisClicked();
    FReply OnRefreshButtonClicked();
    FReply LoadTestResultFromFile();
    void OnGenerate3DMeshStateChanged(ECheckBoxState NewState);

    // HTTP Request Handling
    void TestServerConnection();
    void SendAnalysisRequest();
    void GetAnalysisResult();
    void CheckAnalysisProgress();
    
    // HTTP Response Callbacks
    void OnHttpRequestComplete(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful);
    void OnAnalysisRequestResponse(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful);
    void OnProgressResponse(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful);
    void OnResultResponse(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful);

    // UI State Management
    void UpdateRunButtonState();
    void UpdateStatusText(const FString& Message);
    void UpdateServerStatus(bool bConnected);
    void UpdateProgress(float Progress, const FString& Message);

    // Progress Monitoring
    void StartProgressTimer();
    void StopProgressTimer();
    EActiveTimerReturnType OnProgressTimerTick(double InCurrentTime, float InDeltaTime);

    // Result Processing
    bool ValidateInputFiles();
    void ShowAnalysisResults(const FString& Results);
    void ShowErrorMessage(const FString& ErrorMsg);
    FString GenerateDiagnosticReport(TSharedPtr<FJsonObject> ResultsJson);

    // 3D Visualization
    void Generate3DMeshFromResults(TSharedPtr<FJsonObject> ResultsJson);
    void GenerateTestCube();

    // Slice Viewer Controls
    void ChangeViewPlane(int32 Plane);
    void SyncSliceViewers();

	// Synchronization
	void SetupSliceViewerSync();
	void OnOriginalSliceChanged(int32 NewSlice);
	void OnSegmentationSliceChanged(int32 NewSlice);
	FReply OnToggleSyncClicked();
};