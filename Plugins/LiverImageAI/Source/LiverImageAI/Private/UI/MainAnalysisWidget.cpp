#include "UI/MainAnalysisWidget.h"
#include "Services/AnalysisService.h"
#include "Services/ServerConnectionService.h"
#include "Services/FileService.h"
#include "Rendering/VisualizationService.h"
#include "UI/FileSelectionWidget.h"
#include "UI/ProgressWidget.h"
#include "UI/DiagnosticReportWidget.h"
#include "UI/SliceViewerWidget.h"
#include "Utils/ImageDataProcessor.h"
#include "Utils/JsonParser.h"
#include "Widgets/SBoxPanel.h"  
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Styling/CoreStyle.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Json.h"

#define LOCTEXT_NAMESPACE "SMainAnalysisWidget"

void SMainAnalysisWidget::Construct(const FArguments& InArgs)
{
    // Initialize services
    AnalysisService = MakeShareable(new FAnalysisService());
    ConnectionService = MakeShareable(new FServerConnectionService());
    VisualizationService = MakeShareable(new FVisualizationService());
    
    // Initialize state
    bAnalysisInProgress = false;
    bGenerate3DMesh = true;
    bSliceSyncEnabled = true;
    
    // Setup service callbacks
    ConnectionService->OnConnectionStatusChanged.AddRaw(this, &SMainAnalysisWidget::OnServerConnectionChanged);
    AnalysisService->OnAnalysisStarted.AddRaw(this, &SMainAnalysisWidget::OnAnalysisStarted);
    AnalysisService->OnAnalysisProgress.AddRaw(this, &SMainAnalysisWidget::OnAnalysisProgress);
    AnalysisService->OnAnalysisCompleted.AddRaw(this, &SMainAnalysisWidget::OnAnalysisCompleted);
    AnalysisService->OnAnalysisError.AddRaw(this, &SMainAnalysisWidget::OnAnalysisError);
    
    ChildSlot
    [
        SNew(SVerticalBox)
        
        // Title and server status
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(10)
        [
            CreateServerConnectionSection()
        ]
        
        // File selection section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(10, 5)
        [
            CreateFileSelectionSection()
        ]
        
        // 2D slice viewer section
        + SVerticalBox::Slot()
        .FillHeight(0.6f)
        .Padding(10)
        [
            CreateSliceViewerSection()
        ]
        
        // Control buttons
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(10)
        [
            CreateControlButtonsSection()
        ]
        
        // Diagnostic report section
        + SVerticalBox::Slot()
        .FillHeight(0.35f)
        .Padding(10, 5)
        [
            CreateDiagnosticReportSection()
        ]
        
        // Status bar
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(10, 2)
        [
            SAssignNew(StatusText, STextBlock)
            .Text(FText::FromString(TEXT("Ready")))
            .AutoWrapText(true)
            .Font(FCoreStyle::GetDefaultFontStyle("Regular", 8))
        ]
    ];
    
    SetupSliceViewerSync();
    UpdateUIState();
}

TSharedRef<SWidget> SMainAnalysisWidget::CreateServerConnectionSection()
{
    return SNew(SHorizontalBox)
        + SHorizontalBox::Slot()
        .FillWidth(0.6f)
        [
            SNew(STextBlock)
            .Text(LOCTEXT("PluginTitle", "Liver AI Analysis Plugin"))
            .Font(FCoreStyle::GetDefaultFontStyle("Bold", 16))
            .Justification(ETextJustify::Center)
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.4f)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .FillWidth(0.6f)
            [
                SAssignNew(ServerURLText, SEditableTextBox)
                .Text(FText::FromString(TEXT("http://127.0.0.1:8888")))
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.2f)
            .Padding(5, 0)
            [
                SAssignNew(TestConnectionButton, SButton)
                .Text(LOCTEXT("TestConnection", "Test"))
                .OnClicked(this, &SMainAnalysisWidget::OnTestServerConnectionClicked)
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.2f)
            [
                SAssignNew(ServerStatusText, STextBlock)
                .Text(LOCTEXT("NotConnected", "Not Connected"))
                .ColorAndOpacity(FLinearColor::Yellow)
            ]
        ];
}

TSharedRef<SWidget> SMainAnalysisWidget::CreateFileSelectionSection()
{
    return SNew(SVerticalBox)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SAssignNew(MRIFileSelector, SFileSelectionWidget)
            .Label(TEXT("MRI File:"))
            .FileTypes(TEXT("MRI Files (*.nii;*.nii.gz;*.dcm)|*.nii;*.nii.gz;*.dcm"))
            .HintText(TEXT("Select MRI file..."))
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SAssignNew(LiverModelSelector, SFileSelectionWidget)
            .Label(TEXT("Liver Model:"))
            .FileTypes(TEXT("Model Files (*.pth;*.onnx)|*.pth;*.onnx"))
            .HintText(TEXT("Select liver model..."))
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SAssignNew(VesselModelSelector, SFileSelectionWidget)
            .Label(TEXT("Vessel Model:"))
            .FileTypes(TEXT("Model Files (*.pth;*.onnx)|*.pth;*.onnx"))
            .HintText(TEXT("Select vessel model..."))
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SAssignNew(TumorModelSelector, SFileSelectionWidget)
            .Label(TEXT("Tumor Model:"))
            .FileTypes(TEXT("Model Files (*.pth;*.onnx)|*.pth;*.onnx"))
            .HintText(TEXT("Select tumor model..."))
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 5)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .FillWidth(0.3f)
            [
                SNew(STextBlock)
                .Text(LOCTEXT("Generate3DMesh", "Generate 3D Mesh:"))
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.7f)
            [
                SAssignNew(Generate3DMeshCheckBox, SCheckBox)
                .IsChecked(ECheckBoxState::Checked)
                .OnCheckStateChanged(this, &SMainAnalysisWidget::OnGenerate3DMeshStateChanged)
                [
                    SNew(STextBlock)
                    .Text(LOCTEXT("Generate3DMeshHint", "Automatically generate 3D visualization"))
                ]
            ]
        ];
}

TSharedRef<SWidget> SMainAnalysisWidget::CreateSliceViewerSection()
{
    return SNew(SVerticalBox)
        // View control buttons
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .AutoWidth()
            [
                SNew(STextBlock)
                .Text(LOCTEXT("ViewPlane", "View Plane:"))
            ]
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(5, 0)
            [
                SAssignNew(AxialButton, SButton)
                .Text(LOCTEXT("Axial", "Axial"))
                .OnClicked_Lambda([this]() { ChangeViewPlane(0); return FReply::Handled(); })
            ]
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(5, 0)
            [
                SAssignNew(CoronalButton, SButton)
                .Text(LOCTEXT("Coronal", "Coronal"))
                .OnClicked_Lambda([this]() { ChangeViewPlane(1); return FReply::Handled(); })
            ]
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(5, 0)
            [
                SAssignNew(SagittalButton, SButton)
                .Text(LOCTEXT("Sagittal", "Sagittal"))
                .OnClicked_Lambda([this]() { ChangeViewPlane(2); return FReply::Handled(); })
            ]
        ]
        // Dual canvas layout
        + SVerticalBox::Slot()
        .FillHeight(1.0f)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(5)
            [
                SAssignNew(OriginalSliceViewer, SSliceViewerWidget)
                .Title(TEXT("Original MRI Image"))
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(5)
            [
                SAssignNew(SegmentationSliceViewer, SSliceViewerWidget)
                .Title(TEXT("AI Segmentation Result"))
            ]
        ];
}

TSharedRef<SWidget> SMainAnalysisWidget::CreateControlButtonsSection()
{
    return SNew(SHorizontalBox)
        + SHorizontalBox::Slot()
        .FillWidth(0.2f)
        .Padding(2)
        [
            SAssignNew(RunAnalysisButton, SButton)
            .Text(LOCTEXT("RunAnalysis", "Run AI Analysis"))
            .OnClicked(this, &SMainAnalysisWidget::OnRunAnalysisClicked)
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.2f)
        .Padding(2)
        [
            SAssignNew(CancelAnalysisButton, SButton)
            .Text(LOCTEXT("CancelAnalysis", "Cancel"))
            .OnClicked(this, &SMainAnalysisWidget::OnCancelAnalysisClicked)
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.2f)
        .Padding(2)
        [
            SNew(SButton)
            .Text(LOCTEXT("LoadTest", "Load Test Result"))
            .OnClicked(this, &SMainAnalysisWidget::OnLoadTestResultClicked)
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.2f)
        .Padding(2)
        [
            SNew(SButton)
            .Text(LOCTEXT("ToggleSync", "Toggle Sync"))
            .OnClicked(this, &SMainAnalysisWidget::OnToggleSyncClicked)
        ]
        + SHorizontalBox::Slot()
        .FillWidth(0.2f)
        .Padding(2)
        [
            SAssignNew(ProgressWidget, SProgressWidget)
        ];
}

TSharedRef<SWidget> SMainAnalysisWidget::CreateDiagnosticReportSection()
{
    return SAssignNew(ReportWidget, SDiagnosticReportWidget);
}

// Event handler implementations
FReply SMainAnalysisWidget::OnTestServerConnectionClicked()
{
    FString ServerURL = ServerURLText->GetText().ToString();
    ConnectionService->TestConnection(ServerURL);
    UpdateStatusText(TEXT("Testing server connection..."));
    return FReply::Handled();
}

FReply SMainAnalysisWidget::OnRunAnalysisClicked()
{
    if (!ValidateInputFiles())
    {
        return FReply::Handled();
    }
    
    FLiverAnalysisRequest Request;
    Request.MRIFilePath = MRIFileSelector->GetFilePath();
    Request.LiverModelPath = LiverModelSelector->GetFilePath();
    Request.VesselModelPath = VesselModelSelector->GetFilePath();
    Request.TumorModelPath = TumorModelSelector->GetFilePath();
    Request.RequestId = GenerateRequestId();
    Request.bGenerate3DMesh = bGenerate3DMesh;
    
    AnalysisService->StartAnalysis(Request);
    return FReply::Handled();
}

FReply SMainAnalysisWidget::OnCancelAnalysisClicked()
{
    AnalysisService->CancelAnalysis();
    bAnalysisInProgress = false;
    UpdateUIState();
    UpdateStatusText(TEXT("Analysis cancelled"));
    return FReply::Handled();
}

FReply SMainAnalysisWidget::OnLoadTestResultClicked()
{
    LoadTestResult();
    return FReply::Handled();
}

FReply SMainAnalysisWidget::OnToggleSyncClicked()
{
    bSliceSyncEnabled = !bSliceSyncEnabled;
    SetupSliceViewerSync();
    UpdateStatusText(bSliceSyncEnabled ? TEXT("Slice sync enabled") : TEXT("Slice sync disabled"));
    return FReply::Handled();
}

void SMainAnalysisWidget::OnGenerate3DMeshStateChanged(ECheckBoxState NewState)
{
    bGenerate3DMesh = (NewState == ECheckBoxState::Checked);
}

// Service callbacks
void SMainAnalysisWidget::OnServerConnectionChanged(bool bHealthy, const FString& Status)
{
    if (ServerStatusText.IsValid())
    {
        ServerStatusText->SetText(FText::FromString(bHealthy ? TEXT("Connected") : TEXT("Failed")));
        ServerStatusText->SetColorAndOpacity(bHealthy ? FLinearColor::Green : FLinearColor::Red);
    }
    UpdateStatusText(Status);
    UpdateUIState();
}

void SMainAnalysisWidget::OnAnalysisStarted(const FString& RequestId)
{
    CurrentRequestId = RequestId;
    bAnalysisInProgress = true;
    UpdateUIState();
    UpdateStatusText(FString::Printf(TEXT("Analysis started: %s"), *RequestId));
}

void SMainAnalysisWidget::OnAnalysisProgress(const FAnalysisProgress& Progress)
{
    if (ProgressWidget.IsValid())
    {
        ProgressWidget->UpdateProgress(Progress);
    }
}

void SMainAnalysisWidget::OnAnalysisCompleted(const FLiverAnalysisResult& Result)
{
    bAnalysisInProgress = false;
    UpdateUIState();
    ProcessAnalysisResult(Result);
    UpdateStatusText(TEXT("Analysis completed successfully!"));
}

void SMainAnalysisWidget::OnAnalysisError(const FString& Error)
{
    bAnalysisInProgress = false;
    UpdateUIState();
    UpdateStatusText(FString::Printf(TEXT("Analysis failed: %s"), *Error));
}

// Slice viewer synchronization
void SMainAnalysisWidget::SetupSliceViewerSync()
{
    if (bSliceSyncEnabled)
    {
        if (OriginalSliceViewer.IsValid())
        {
            OriginalSliceViewer->OnSliceChanged.BindRaw(this, &SMainAnalysisWidget::OnOriginalSliceChanged);
        }
        
        if (SegmentationSliceViewer.IsValid())
        {
            SegmentationSliceViewer->OnSliceChanged.BindRaw(this, &SMainAnalysisWidget::OnSegmentationSliceChanged);
        }
    }
    else
    {
        if (OriginalSliceViewer.IsValid())
        {
            OriginalSliceViewer->OnSliceChanged.Unbind();
        }
        
        if (SegmentationSliceViewer.IsValid())
        {
            SegmentationSliceViewer->OnSliceChanged.Unbind();
        }
    }
}

void SMainAnalysisWidget::OnOriginalSliceChanged(int32 NewSlice)
{
    if (bSliceSyncEnabled && SegmentationSliceViewer.IsValid())
    {
        SegmentationSliceViewer->OnSliceChanged.Unbind();
        SegmentationSliceViewer->SetCurrentSlice(NewSlice);
        SegmentationSliceViewer->OnSliceChanged.BindRaw(this, &SMainAnalysisWidget::OnSegmentationSliceChanged);
    }
}

void SMainAnalysisWidget::OnSegmentationSliceChanged(int32 NewSlice)
{
    if (bSliceSyncEnabled && OriginalSliceViewer.IsValid())
    {
        OriginalSliceViewer->OnSliceChanged.Unbind();
        OriginalSliceViewer->SetCurrentSlice(NewSlice);
        OriginalSliceViewer->OnSliceChanged.BindRaw(this, &SMainAnalysisWidget::OnOriginalSliceChanged);
    }
}

void SMainAnalysisWidget::ChangeViewPlane(int32 Plane)
{
    if (OriginalSliceViewer.IsValid())
    {
        OriginalSliceViewer->SetViewPlane(Plane);
    }
    
    if (SegmentationSliceViewer.IsValid())
    {
        SegmentationSliceViewer->SetViewPlane(Plane);
    }
    
    SetupSliceViewerSync();
}

// Helper methods
bool SMainAnalysisWidget::ValidateInputFiles()
{
    if (!MRIFileSelector->IsPathValid())
    {
        UpdateStatusText(TEXT("Please select a valid MRI file"));
        return false;
    }
    
    if (!LiverModelSelector->IsPathValid())
    {
        UpdateStatusText(TEXT("Please select a valid liver model"));
        return false;
    }
    
    if (!VesselModelSelector->IsPathValid())
    {
        UpdateStatusText(TEXT("Please select a valid vessel model"));
        return false;
    }
    
    if (!TumorModelSelector->IsPathValid())
    {
        UpdateStatusText(TEXT("Please select a valid tumor model"));
        return false;
    }
    
    return true;
}

FString SMainAnalysisWidget::GenerateRequestId()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("ue_request_%s"), *Now.ToString(TEXT("%Y%m%d_%H%M%S")));
}

void SMainAnalysisWidget::UpdateUIState()
{
    bool bCanRunAnalysis = ConnectionService->IsServerHealthy() && 
                          !bAnalysisInProgress && 
                          ValidateInputFiles();
    
    if (RunAnalysisButton.IsValid())
    {
        RunAnalysisButton->SetEnabled(bCanRunAnalysis);
    }
    
    if (CancelAnalysisButton.IsValid())
    {
        CancelAnalysisButton->SetEnabled(bAnalysisInProgress);
    }
}

void SMainAnalysisWidget::UpdateStatusText(const FString& Message)
{
    if (StatusText.IsValid())
    {
        StatusText->SetText(FText::FromString(Message));
    }
    
    UE_LOG(LogLiverAI, Log, TEXT("Status: %s"), *Message);
}

void SMainAnalysisWidget::ProcessAnalysisResult(const FLiverAnalysisResult& Result)
{
    // Display diagnostic report
    if (ReportWidget.IsValid())
    {
        ReportWidget->ShowReport(Result);
    }
    
    // TODO: Process image data and update slice viewers
    // This will depend on how the result contains image data
    
    // Generate 3D visualization if enabled and mesh data exists
    if (bGenerate3DMesh && Result.MeshData.Num() > 0)
    {
        VisualizationService->GenerateMeshFromData(Result.MeshData, Result.RequestId);
    }
}

void SMainAnalysisWidget::LoadTestResult()
{
    FString TestFilePath = TEXT("D:/Project/TD/multiliver_ai/test_results/analysis_result_test_20250919_195142.json");
    FString FileContent;
    
    if (FFileHelper::LoadFileToString(FileContent, *TestFilePath))
    {
        UpdateStatusText(TEXT("Test result loaded successfully"));
        
        TSharedPtr<FJsonObject> JsonObject;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileContent);
        
        if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
        {
            FJsonParser Parser;
            FLiverAnalysisResult Result = Parser.ParseAnalysisResult(FileContent);
            
            // Check for Base64 encoded image data
            FString ImageDataBase64, SegmentationDataBase64;
            if (JsonObject->TryGetStringField(TEXT("image_data_base64"), ImageDataBase64) &&
                JsonObject->TryGetStringField(TEXT("segmentation_data_base64"), SegmentationDataBase64))
            {
                // Convert Base64 to data arrays
                TArray<float> ImageData = FImageDataProcessor::ConvertBase64ToFloatArray(ImageDataBase64);
                TArray<uint8> SegmentationData = FImageDataProcessor::ConvertBase64ToByteArray(SegmentationDataBase64);
                
                // Get dimensions
                TArray<int32> ImageDimensions;
                const TArray<TSharedPtr<FJsonValue>>* DimsArray;
                if (JsonObject->TryGetArrayField(TEXT("image_dimensions"), DimsArray))
                {
                    for (const auto& Dim : *DimsArray)
                    {
                        ImageDimensions.Add(Dim->AsNumber());
                    }
                }
                
                if (ImageDimensions.Num() == 3)
                {
                    // Update slice viewers
                    if (OriginalSliceViewer.IsValid())
                    {
                        OriginalSliceViewer->SetImageData(ImageData, 
                            ImageDimensions[0], ImageDimensions[1], ImageDimensions[2]);
                    }
                    
                    if (SegmentationSliceViewer.IsValid())
                    {
                        SegmentationSliceViewer->SetImageData(ImageData,
                            ImageDimensions[0], ImageDimensions[1], ImageDimensions[2]);
                        SegmentationSliceViewer->SetMaskData(SegmentationData,
                            ImageDimensions[0], ImageDimensions[1], ImageDimensions[2]);
                    }
                }
            }
            
            ProcessAnalysisResult(Result);
        }
    }
    else
    {
        UpdateStatusText(TEXT("Failed to load test file"));
    }
}

#undef LOCTEXT_NAMESPACE