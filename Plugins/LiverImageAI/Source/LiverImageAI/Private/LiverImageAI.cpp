#include "LiverImageAI.h"
#include "LiverAITypes.h"
#include "ToolMenus.h"
#include "Framework/Docking/TabManager.h"
#include "Widgets/Docking/SDockTab.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SCheckBox.h"
#include "Framework/Application/SlateApplication.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "IDesktopPlatform.h"
#include "DesktopPlatformModule.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Misc/DateTime.h"
#include "Misc/Base64.h"
#include "Editor.h"
#include "Engine/World.h"
#include "Components/SceneComponent.h"
#include "ProceduralMeshComponent.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "SLiverSliceViewer.h"
#include "Widgets/Input/SMultiLineEditableTextBox.h"
#include "UObject/ConstructorHelpers.h"

// Define log category
DEFINE_LOG_CATEGORY(LogLiverAI);

static const FName LiverAITabName("LiverImageAI");

#define LOCTEXT_NAMESPACE "FLiverImageAIModule"

void FLiverImageAIModule::StartupModule()
{
	FLiverImageAICommands::Register();
	
	PluginCommands = MakeShareable(new FUICommandList);
	PluginCommands->MapAction(
		FLiverImageAICommands::Get().OpenPluginWindow,
		FExecuteAction::CreateRaw(this, &FLiverImageAIModule::PluginButtonClicked),
		FCanExecuteAction());

	UToolMenus::RegisterStartupCallback(FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FLiverImageAIModule::RegisterMenus));
	
	FGlobalTabmanager::Get()->RegisterNomadTabSpawner(LiverAITabName,
		FOnSpawnTab::CreateRaw(this, &FLiverImageAIModule::OnSpawnPluginTab))
		.SetDisplayName(LOCTEXT("FLiverAITabTitle", "Liver AI Analysis"));

	UE_LOG(LogLiverAI, Log, TEXT("Liver AI Plugin started"));
}

void FLiverImageAIModule::ShutdownModule()
{
	UToolMenus::UnRegisterStartupCallback(this);
	UToolMenus::UnregisterOwner(this);
	FLiverImageAICommands::Unregister();
	FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(LiverAITabName);

	UE_LOG(LogLiverAI, Log, TEXT("Liver AI Plugin shut down"));
}

TSharedRef<SDockTab> FLiverImageAIModule::OnSpawnPluginTab(const FSpawnTabArgs& SpawnTabArgs)
{
	return SNew(SDockTab)
		.TabRole(ETabRole::NomadTab)
		[
			SNew(SLiverAIWidget)
		];
}

void FLiverImageAIModule::PluginButtonClicked()
{
	FGlobalTabmanager::Get()->TryInvokeTab(LiverAITabName);
}

void FLiverImageAIModule::RegisterMenus()
{
	FToolMenuOwnerScoped OwnerScoped(this);

	{
		UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Window");
		{
			FToolMenuSection& Section = Menu->FindOrAddSection("WindowLayout");
			Section.AddMenuEntryWithCommandList(FLiverImageAICommands::Get().OpenPluginWindow, PluginCommands);
		}
	}
}

void FLiverImageAICommands::RegisterCommands()
{
	UI_COMMAND(OpenPluginWindow, "Liver AI", "Open Liver AI Analysis window", EUserInterfaceActionType::Button, FInputChord());
}

void SLiverAIWidget::Construct(const FArguments& InArgs)
{
	// Initialize state variables
	bServerHealthy = false;
	bAnalysisInProgress = false;
	bGenerate3DMesh = true;
	ServerURL = TEXT("http://127.0.0.1:8888");
	CurrentRequestId = TEXT("");
	CurrentProgress = 0.0f;
	LastProgressCheckTime = 0.0f;

	ChildSlot
	[
		SNew(SVerticalBox)
		
		// Title and server status
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10)
		[
			SNew(SHorizontalBox)
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
				SAssignNew(ServerStatusText, STextBlock)
				.Text(LOCTEXT("ServerStatus", "Server: Not Tested"))
				.ColorAndOpacity(FLinearColor::Yellow)
				.Justification(ETextJustify::Right)
			]
		]

		// Server address configuration
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.2f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("ServerURLLabel", "Server Address:"))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.6f)
			[
				SAssignNew(ServerURLText, SEditableTextBox)
				.Text(FText::FromString(ServerURL))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.2f)
			[
				SNew(SButton)
				.Text(LOCTEXT("TestConnectionButton", "Test Connection"))
				.OnClicked(this, &SLiverAIWidget::OnTestServerConnectionClicked)
			]
		]
		
		// MRI file selection
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("MRIFileLabel", "MRI File:"))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SAssignNew(MRIFilePathText, SEditableTextBox)
				.HintText(LOCTEXT("MRIFileHint", "Select MRI file..."))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.2f)
			[
				SNew(SButton)
				.Text(LOCTEXT("BrowseButton", "Browse"))
				.OnClicked(this, &SLiverAIWidget::OnBrowseMRIFileClicked)
			]
		]
		
		// Liver model selection
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("LiverModelLabel", "Liver Model:"))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SAssignNew(LiverModelPathText, SEditableTextBox)
				.HintText(LOCTEXT("LiverModelHint", "Select liver model..."))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.2f)
			[
				SNew(SButton)
				.Text(LOCTEXT("BrowseButton", "Browse"))
				.OnClicked(this, &SLiverAIWidget::OnBrowseLiverModelClicked)
			]
		]
		
		// Vessel model selection
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("VesselModelLabel", "Vessel Model:"))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SAssignNew(VesselModelPathText, SEditableTextBox)
				.HintText(LOCTEXT("VesselModelHint", "Select vessel model..."))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.2f)
			[
				SNew(SButton)
				.Text(LOCTEXT("BrowseButton", "Browse"))
				.OnClicked(this, &SLiverAIWidget::OnBrowseVesselModelClicked)
			]
		]
		
		// Tumor model selection
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("TumorModelLabel", "Tumor Model:"))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SAssignNew(TumorModelPathText, SEditableTextBox)
				.HintText(LOCTEXT("TumorModelHint", "Select tumor model..."))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.2f)
			[
				SNew(SButton)
				.Text(LOCTEXT("BrowseButton", "Browse"))
				.OnClicked(this, &SLiverAIWidget::OnBrowseTumorModelClicked)
			]
		]
		
		// 3D mesh generation option
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("Generate3DMeshLabel", "Generate 3D Mesh:"))
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.7f)
			[
				SAssignNew(Generate3DMeshCheckBox, SCheckBox)
				.IsChecked(ECheckBoxState::Checked)
				.OnCheckStateChanged(this, &SLiverAIWidget::OnGenerate3DMeshStateChanged)
				[
					SNew(STextBlock)
					.Text(LOCTEXT("Generate3DMeshHint", "Automatically generate 3D visualization model after analysis"))
				]
			]
		]

		// 2D image viewer section - maintain square aspect ratio
		+ SVerticalBox::Slot()
		.FillHeight(0.6f)
		.Padding(10)
		[
			SNew(SVerticalBox)
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
					.Text(LOCTEXT("ViewPlaneLabel", "View Plane:"))
				]
				+ SHorizontalBox::Slot()
				.AutoWidth()
				.Padding(5, 0)
				[
					SAssignNew(AxialButton, SButton)
					.Text(LOCTEXT("AxialView", "Axial"))
					.OnClicked_Lambda([this]() { ChangeViewPlane(0); return FReply::Handled(); })
				]
				+ SHorizontalBox::Slot()
				.AutoWidth()
				.Padding(5, 0)
				[
					SAssignNew(CoronalButton, SButton)
					.Text(LOCTEXT("CoronalView", "Coronal"))
					.OnClicked_Lambda([this]() { ChangeViewPlane(1); return FReply::Handled(); })
				]
				+ SHorizontalBox::Slot()
				.AutoWidth()
				.Padding(5, 0)
				[
					SAssignNew(SagittalButton, SButton)
					.Text(LOCTEXT("SagittalView", "Sagittal"))
					.OnClicked_Lambda([this]() { ChangeViewPlane(2); return FReply::Handled(); })
				]
				+ SHorizontalBox::Slot()
				.FillWidth(1.0f)
				[
					SNew(SSpacer)
				]
			]
			// Dual canvas layout - add square constraint
			+ SVerticalBox::Slot()
			.FillHeight(1.0f)
			[
				SNew(SHorizontalBox)
				// Original image viewer - use AspectRatio to maintain square
				+ SHorizontalBox::Slot()
				.FillWidth(0.5f)
				.Padding(5)
				[
					SNew(SBorder)
					.BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.GroupBorder"))
					[
						SNew(SVerticalBox)
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(5)
						[
							SNew(STextBlock)
							.Text(LOCTEXT("OriginalImageLabel", "Original MRI Image"))
							.Font(FCoreStyle::GetDefaultFontStyle("Bold", 11))
							.Justification(ETextJustify::Center)
						]
						+ SVerticalBox::Slot()
						.FillHeight(1.0f)
						.HAlign(HAlign_Center)
						.VAlign(VAlign_Center)
						[
							// Use AspectRatio to keep image square
							SNew(SBox)
							.WidthOverride(500.0f)
							.HeightOverride(500.0f)
							[
								SAssignNew(OriginalSliceViewer, SLiverSliceViewer)
							]
						]
					]
				]
				// Segmentation result viewer - also maintain square
				+ SHorizontalBox::Slot()
				.FillWidth(0.5f)
				.Padding(5)
				[
					SNew(SBorder)
					.BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.GroupBorder"))
					[
						SNew(SVerticalBox)
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(5)
						[
							SNew(STextBlock)
							.Text(LOCTEXT("SegmentationLabel", "AI Segmentation Result"))
							.Font(FCoreStyle::GetDefaultFontStyle("Bold", 11))
							.Justification(ETextJustify::Center)
						]
						+ SVerticalBox::Slot()
						.FillHeight(1.0f)
						.HAlign(HAlign_Center)
						.VAlign(VAlign_Center)
						[
							// Use AspectRatio to keep image square
							SNew(SBox)
							.WidthOverride(500.0f)
							.HeightOverride(500.0f)
							[
								SAssignNew(SegmentationSliceViewer, SLiverSliceViewer)
							]
						]
					]
				]
			]
		]

		// Control buttons
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.25f)
			[
				SAssignNew(RunAnalysisButton, SButton)
				.Text(LOCTEXT("RunAnalysisButton", "Run AI Analysis"))
				.IsEnabled(false)
				.OnClicked(this, &SLiverAIWidget::OnRunAnalysisClicked)
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.25f)
			[
				SAssignNew(CancelAnalysisButton, SButton)
				.Text(LOCTEXT("CancelAnalysisButton", "Cancel Analysis"))
				.IsEnabled(false)
				.OnClicked(this, &SLiverAIWidget::OnCancelAnalysisClicked)
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.25f)
			[
				SNew(SButton)
				.Text(LOCTEXT("LoadTestResultButton", "Load Test Result"))
				.OnClicked(this, &SLiverAIWidget::LoadTestResultFromFile)
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.25f)
			[
				SNew(SButton)
				.Text(LOCTEXT("SyncToggleButton", "Toggle Sync"))
				.OnClicked(this, &SLiverAIWidget::OnToggleSyncClicked)
			]
		]

		// Diagnostic results text area - fix scroll wheel support
		+ SVerticalBox::Slot()
		.FillHeight(0.35f)
		.Padding(10, 5)
		[
			SNew(SBorder)
			.BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.GroupBorder"))
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(5)
				[
					SNew(SHorizontalBox)
					+ SHorizontalBox::Slot()
					.AutoWidth()
					[
						SNew(STextBlock)
						.Text(LOCTEXT("DiagnosticResultsLabel", "AI Diagnostic Report"))
						.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
					]
					+ SHorizontalBox::Slot()
					.FillWidth(1.0f)
					[
						SNew(SSpacer)
					]
					+ SHorizontalBox::Slot()
					.AutoWidth()
					[
						SAssignNew(ProgressText, STextBlock)
						.Text(LOCTEXT("ProgressStatus", "Progress: 0% - Ready"))
						.ColorAndOpacity(FLinearColor::White)
						.Font(FCoreStyle::GetDefaultFontStyle("Regular", 9))
					]
				]
				+ SVerticalBox::Slot()
				.FillHeight(1.0f)
				.Padding(5)
				[
					// Fix scroll wheel issue: use SMultiLineEditableTextBox's built-in scrolling
					SAssignNew(DiagnosticResultsText, SMultiLineEditableTextBox)
					.IsReadOnly(true)
					.Text(LOCTEXT("NoResultsYet", "No analysis results yet. Please load a MRI file and run AI analysis to view diagnostic report."))
					.Font(FCoreStyle::GetDefaultFontStyle("Mono", 8))
					.AllowContextMenu(false)
					.AlwaysShowScrollbars(true)
					.AutoWrapText(true)
				]
			]
		]
		
		// Bottom status display
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(10, 2)
		[
			SAssignNew(StatusText, STextBlock)
			.Text(LOCTEXT("ReadyStatus", "Ready - Please test server connection first, then select required files"))
			.AutoWrapText(true)
			.Font(FCoreStyle::GetDefaultFontStyle("Regular", 8))
		]
	];

	// Setup synchronization callback
	SetupSliceViewerSync();
	
	// Initial state update
	UpdateRunButtonState();
}
// New method: Setup slice viewer synchronization
void SLiverAIWidget::SetupSliceViewerSync()
{
	UE_LOG(LogLiverAI, Warning, TEXT("Setting up slice viewer synchronization"));
	
	// Setup sync callback for original image viewer
	if (OriginalSliceViewer.IsValid())
	{
		OriginalSliceViewer->OnSliceChanged.BindSP(this, &SLiverAIWidget::OnOriginalSliceChanged);
	}
	
	// Setup sync callback for segmentation result viewer
	if (SegmentationSliceViewer.IsValid())
	{
		SegmentationSliceViewer->OnSliceChanged.BindSP(this, &SLiverAIWidget::OnSegmentationSliceChanged);
	}
}

// New method: Original image slice changed callback
void SLiverAIWidget::OnOriginalSliceChanged(int32 NewSlice)
{
	UE_LOG(LogLiverAI, Warning, TEXT("Original slice changed to: %d"), NewSlice);
	
	// Synchronize to segmentation result viewer
	if (SegmentationSliceViewer.IsValid())
	{
		// Temporarily unbind callback to avoid loop
		SegmentationSliceViewer->OnSliceChanged.Unbind();
		SegmentationSliceViewer->SetCurrentSlice(NewSlice);
		// Re-bind callback
		SegmentationSliceViewer->OnSliceChanged.BindSP(this, &SLiverAIWidget::OnSegmentationSliceChanged);
	}
}

// New method: Segmentation result slice changed callback
void SLiverAIWidget::OnSegmentationSliceChanged(int32 NewSlice)
{
	UE_LOG(LogLiverAI, Warning, TEXT("Segmentation slice changed to: %d"), NewSlice);
	
	// Synchronize to original image viewer
	if (OriginalSliceViewer.IsValid())
	{
		// Temporarily unbind callback to avoid loop
		OriginalSliceViewer->OnSliceChanged.Unbind();
		OriginalSliceViewer->SetCurrentSlice(NewSlice);
		// Re-bind callback
		OriginalSliceViewer->OnSliceChanged.BindSP(this, &SLiverAIWidget::OnOriginalSliceChanged);
	}
}

// New method: Toggle sync mode button
FReply SLiverAIWidget::OnToggleSyncClicked()
{
	static bool bSyncEnabled = true;
	bSyncEnabled = !bSyncEnabled;
	
	if (bSyncEnabled)
	{
		SetupSliceViewerSync();
		UpdateStatusText(TEXT("Slice Sync: Enabled"));
	}
	else
	{
		// Disable synchronization
		if (OriginalSliceViewer.IsValid())
		{
			OriginalSliceViewer->OnSliceChanged.Unbind();
		}
		if (SegmentationSliceViewer.IsValid())
		{
			SegmentationSliceViewer->OnSliceChanged.Unbind();
		}
		UpdateStatusText(TEXT("Slice Sync: Disabled"));
	}
	
	return FReply::Handled();
}

// Modified method: Re-setup sync when changing view plane
void SLiverAIWidget::ChangeViewPlane(int32 Plane)
{
    if (OriginalSliceViewer.IsValid())
    {
        OriginalSliceViewer->SetViewPlane(Plane);
    }
    if (SegmentationSliceViewer.IsValid())
    {
        SegmentationSliceViewer->SetViewPlane(Plane);
    }
    
    // Re-setup sync and synchronize slice position
    SetupSliceViewerSync();
    SyncSliceViewers();
}

// Keep existing sync method as backup
void SLiverAIWidget::SyncSliceViewers()
{
    if (OriginalSliceViewer.IsValid() && SegmentationSliceViewer.IsValid())
    {
        int32 CurrentSlice = OriginalSliceViewer->GetCurrentSlice();
        SegmentationSliceViewer->SetCurrentSlice(CurrentSlice);
    }
}


bool SLiverAIWidget::OpenFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename)
{
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		TArray<FString> OpenFilenames;
		bool bOpened = DesktopPlatform->OpenFileDialog(
			FSlateApplication::Get().FindBestParentWindowHandleForDialogs(AsShared()),
			Title,
			TEXT(""),
			TEXT(""),
			FileTypes,
			EFileDialogFlags::None,
			OpenFilenames
		);

		if (bOpened && OpenFilenames.Num() > 0)
		{
			OutFilename = OpenFilenames[0];
			return true;
		}
	}
	return false;
}

// Send analysis request to Python backend
void SLiverAIWidget::SendAnalysisRequest()
{
	if (!FHttpModule::Get().IsHttpEnabled())
	{
		UpdateStatusText(TEXT("Error: HTTP module not enabled"));
		OnCancelAnalysisClicked();
		return;
	}

	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
	HttpRequest->OnProcessRequestComplete().BindSP(this, &SLiverAIWidget::OnAnalysisRequestResponse);
	HttpRequest->SetVerb("POST");
	HttpRequest->SetURL(ServerURL + TEXT("/api/analyze"));
	HttpRequest->SetHeader("Content-Type", TEXT("application/json"));
	HttpRequest->SetTimeout(300.0f); // 5 minute timeout

	// Build JSON request data
	TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
	JsonObject->SetStringField(TEXT("mri_file_path"), MRIFilePathText->GetText().ToString());
	JsonObject->SetStringField(TEXT("liver_model_path"), LiverModelPathText->GetText().ToString());
	JsonObject->SetStringField(TEXT("vessel_model_path"), VesselModelPathText->GetText().ToString());
	JsonObject->SetStringField(TEXT("tumor_model_path"), TumorModelPathText->GetText().ToString());
	JsonObject->SetStringField(TEXT("request_id"), CurrentRequestId);
	JsonObject->SetBoolField(TEXT("generate_3d_mesh"), bGenerate3DMesh);

	FString OutputString;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
	FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);
	HttpRequest->SetContentAsString(OutputString);

	if (!HttpRequest->ProcessRequest())
	{
		UpdateStatusText(TEXT("Error: Failed to send analysis request"));
		OnCancelAnalysisClicked();
	}
	else
	{
		UpdateProgress(5.0f, TEXT("Analysis request sent, waiting for server response..."));
	}
}

// Validate input files
bool SLiverAIWidget::ValidateInputFiles()
{
	FString MRIPath = MRIFilePathText->GetText().ToString();
	FString LiverPath = LiverModelPathText->GetText().ToString();
	FString VesselPath = VesselModelPathText->GetText().ToString();
	FString TumorPath = TumorModelPathText->GetText().ToString();
	
	if (MRIPath.IsEmpty())
	{
		UpdateStatusText(TEXT("Error: Please select MRI file"));
		return false;
	}
	
	if (LiverPath.IsEmpty())
	{
		UpdateStatusText(TEXT("Error: Please select liver segmentation model"));
		return false;
	}
	
	if (VesselPath.IsEmpty())
	{
		UpdateStatusText(TEXT("Error: Please select vessel segmentation model"));
		return false;
	}
	
	if (TumorPath.IsEmpty())
	{
		UpdateStatusText(TEXT("Error: Please select tumor segmentation model"));
		return false;
	}
	
	// Check if files exist
	if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*MRIPath))
	{
		UpdateStatusText(TEXT("Error: MRI file does not exist, please reselect"));
		return false;
	}
	
	if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*LiverPath))
	{
		UpdateStatusText(TEXT("Error: Liver model file does not exist, please reselect"));
		return false;
	}
	
	if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*VesselPath))
	{
		UpdateStatusText(TEXT("Error: Vessel model file does not exist, please reselect"));
		return false;
	}
	
	if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*TumorPath))
	{
		UpdateStatusText(TEXT("Error: Tumor model file does not exist, please reselect"));
		return false;
	}
	
	return true;
}

FReply SLiverAIWidget::OnRefreshButtonClicked()
{
	UpdateStatusText(TEXT("Refresh function clicked"));
	return FReply::Handled();
}

// Load test results from file
FReply SLiverAIWidget::LoadTestResultFromFile()
{
	// Directly load test file
	FString TestFilePath = TEXT("D:/Project/TD/multiliver_ai/test_results/analysis_result_test_20250919_195142.json");
	FString FileContent;
	
	if (FFileHelper::LoadFileToString(FileContent, *TestFilePath))
	{
		UE_LOG(LogLiverAI, Warning, TEXT("Loaded test file: %s"), *TestFilePath);
		UpdateStatusText(TEXT("Test result file loaded successfully"));
		bGenerate3DMesh = true; // Ensure 3D generation is enabled
		ShowAnalysisResults(FileContent);
	}
	else
	{
		UpdateStatusText(TEXT("Unable to load test file"));
	}
	
	return FReply::Handled();
}

FReply SLiverAIWidget::OnBrowseMRIFileClicked()
{
	FString Filename;
	if (OpenFileDialog(TEXT("Select MRI File"), TEXT("MRI Files (*.nii;*.nii.gz;*.dcm)|*.nii;*.nii.gz;*.dcm"), Filename))
	{
		MRIFilePathText->SetText(FText::FromString(Filename));
		UpdateStatusText(FString::Printf(TEXT("MRI file selected: %s"), *FPaths::GetCleanFilename(Filename)));
		UpdateRunButtonState();
	}
	else
	{
		UpdateStatusText(TEXT("MRI file selection cancelled"));
	}
	return FReply::Handled();
}

FReply SLiverAIWidget::OnBrowseLiverModelClicked()
{
	FString Filename;
	if (OpenFileDialog(TEXT("Select Liver Segmentation Model"), TEXT("Model Files (*.pth;*.onnx)|*.pth;*.onnx"), Filename))
	{
		LiverModelPathText->SetText(FText::FromString(Filename));
		UpdateStatusText(FString::Printf(TEXT("Liver model selected: %s"), *FPaths::GetCleanFilename(Filename)));
		UpdateRunButtonState();
	}
	else
	{
		UpdateStatusText(TEXT("Liver model selection cancelled"));
	}
	return FReply::Handled();
}

FReply SLiverAIWidget::OnBrowseVesselModelClicked()
{
	FString Filename;
	if (OpenFileDialog(TEXT("Select Vessel Segmentation Model"), TEXT("Model Files (*.pth;*.onnx)|*.pth;*.onnx"), Filename))
	{
		VesselModelPathText->SetText(FText::FromString(Filename));
		UpdateStatusText(FString::Printf(TEXT("Vessel model selected: %s"), *FPaths::GetCleanFilename(Filename)));
		UpdateRunButtonState();
	}
	else
	{
		UpdateStatusText(TEXT("Vessel model selection cancelled"));
	}
	return FReply::Handled();
}

FReply SLiverAIWidget::OnBrowseTumorModelClicked()
{
	FString Filename;
	if (OpenFileDialog(TEXT("Select Tumor Segmentation Model"), TEXT("Model Files (*.pth;*.onnx)|*.pth;*.onnx"), Filename))
	{
		TumorModelPathText->SetText(FText::FromString(Filename));
		UpdateStatusText(FString::Printf(TEXT("Tumor model selected: %s"), *FPaths::GetCleanFilename(Filename)));
		UpdateRunButtonState();
	}
	else
	{
		UpdateStatusText(TEXT("Tumor model selection cancelled"));
	}
	return FReply::Handled();
}

FReply SLiverAIWidget::OnRunAnalysisClicked()
{
	if (bAnalysisInProgress)
	{
		return FReply::Handled();
	}
	
	// Validate input files
	if (!ValidateInputFiles())
	{
		return FReply::Handled();
	}
	
	// Generate request ID
	FDateTime Now = FDateTime::Now();
	CurrentRequestId = FString::Printf(TEXT("ue_request_%s"), *Now.ToString(TEXT("%Y%m%d_%H%M%S")));
	
	// Update UI state
	bAnalysisInProgress = true;
	CurrentProgress = 0.0f;
	UpdateRunButtonState();
	UpdateProgress(0.0f, TEXT("Preparing to send analysis request..."));
	UpdateStatusText(FString::Printf(TEXT("Starting AI analysis - Request ID: %s"), *CurrentRequestId));
	
	// Send analysis request
	SendAnalysisRequest();
	
	return FReply::Handled();
}

FReply SLiverAIWidget::OnCancelAnalysisClicked()
{
	if (bAnalysisInProgress)
	{
		StopProgressTimer();
		bAnalysisInProgress = false;
		CurrentRequestId = TEXT("");
		CurrentProgress = 0.0f;
		UpdateRunButtonState();
		UpdateProgress(0.0f, TEXT("Analysis cancelled"));
		UpdateStatusText(TEXT("User cancelled AI analysis operation"));
	}
	return FReply::Handled();
}

FReply SLiverAIWidget::OnTestServerConnectionClicked()
{
	ServerURL = ServerURLText->GetText().ToString();
	UpdateStatusText(FString::Printf(TEXT("Testing server connection: %s"), *ServerURL));
	TestServerConnection();
	return FReply::Handled();
}

void SLiverAIWidget::TestServerConnection()
{
	if (!FHttpModule::Get().IsHttpEnabled())
	{
		UpdateStatusText(TEXT("Error: HTTP module not enabled"));
		UpdateServerStatus(false);
		return;
	}

	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
	HttpRequest->OnProcessRequestComplete().BindSP(this, &SLiverAIWidget::OnHttpRequestComplete);
	HttpRequest->SetVerb("GET");
	HttpRequest->SetURL(ServerURL + TEXT("/api/health"));
	HttpRequest->SetTimeout(10.0f);

	if (!HttpRequest->ProcessRequest())
	{
		UpdateStatusText(TEXT("Error: Cannot send HTTP request"));
		UpdateServerStatus(false);
	}
	else
	{
		UpdateStatusText(TEXT("Connecting to server..."));
	}
}
// Analysis request response handler
void SLiverAIWidget::OnAnalysisRequestResponse(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful)
{
	if (!bAnalysisInProgress)
	{
		return; // User cancelled
	}

	if (!bWasSuccessful || !Response.IsValid())
	{
		UpdateStatusText(TEXT("Analysis request failed: Network connection issue"));
		ShowErrorMessage(TEXT("Unable to connect to server"));
		OnCancelAnalysisClicked();
		return;
	}

	int32 ResponseCode = Response->GetResponseCode();
	FString ResponseContent = Response->GetContentAsString();

	if (ResponseCode == 200)
	{
		UpdateProgress(10.0f, TEXT("Server accepted analysis request"));
		UpdateStatusText(FString::Printf(TEXT("Analysis request successfully submitted! Request ID: %s"), *CurrentRequestId));
		
		// Check if response already contains complete result
		TSharedPtr<FJsonObject> JsonObject;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseContent);
		
		if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
		{
			// If response already contains complete result
			if (JsonObject->HasField(TEXT("organ_stats")) || JsonObject->HasField(TEXT("mesh_data")))
			{
				UE_LOG(LogLiverAI, Warning, TEXT("Server returned complete result immediately"));
				bAnalysisInProgress = false;
				UpdateRunButtonState();
				UpdateProgress(100.0f, TEXT("Analysis complete!"));
				ShowAnalysisResults(ResponseContent);
				return;
			}
		}
		
		// Otherwise start progress polling
		StartProgressTimer();
		
		// Also attempt to get result directly (as backup)
		FTimerHandle DelayedResultTimer;
		if (GEditor)
		{
			GEditor->GetTimerManager()->SetTimer(
				DelayedResultTimer,
				[this]()
				{
					if (bAnalysisInProgress)
					{
						UE_LOG(LogLiverAI, Warning, TEXT("Attempting direct result fetch..."));
						GetAnalysisResult();
					}
				},
				5.0f, // Try to get result directly after 5 seconds
				false
			);
		}
	}
	else
	{
		FString ErrorMsg = FString::Printf(TEXT("Server error (code: %d)"), ResponseCode);
		UpdateStatusText(FString::Printf(TEXT("Analysis request failed: %s"), *ErrorMsg));
		ShowErrorMessage(ErrorMsg);
		OnCancelAnalysisClicked();
	}
}

// Progress polling response handler
void SLiverAIWidget::OnProgressResponse(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful)
{
	if (!bAnalysisInProgress)
	{
		return; // User cancelled
	}

	if (bWasSuccessful && Response.IsValid() && Response->GetResponseCode() == 200)
	{
		FString ResponseContent = Response->GetContentAsString();
		UE_LOG(LogLiverAI, Warning, TEXT("Progress response: %s"), *ResponseContent.Left(500));
		
		TSharedPtr<FJsonObject> JsonObject;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseContent);
		
		if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
		{
			// Try different field names
			FString Status;
			if (JsonObject->HasField(TEXT("status")))
			{
				Status = JsonObject->GetStringField(TEXT("status"));
			}
			else if (JsonObject->HasField(TEXT("state")))
			{
				Status = JsonObject->GetStringField(TEXT("state"));
			}
			
			float Progress = 0.0f;
			if (JsonObject->HasField(TEXT("progress")))
			{
				Progress = JsonObject->GetNumberField(TEXT("progress"));
			}
			else if (JsonObject->HasField(TEXT("percent")))
			{
				Progress = JsonObject->GetNumberField(TEXT("percent"));
			}
			
			FString Message = TEXT("");
			if (JsonObject->HasField(TEXT("message")))
			{
				Message = JsonObject->GetStringField(TEXT("message"));
			}
			else if (JsonObject->HasField(TEXT("status_message")))
			{
				Message = JsonObject->GetStringField(TEXT("status_message"));
			}
			
			UpdateProgress(Progress, Message.IsEmpty() ? Status : Message);
			
			// Check if completed
			if (Status == TEXT("completed") || Status == TEXT("complete") || Status == TEXT("finished") || Progress >= 100.0f)
			{
				UE_LOG(LogLiverAI, Warning, TEXT("Analysis completed! Getting final result..."));
				StopProgressTimer();
				GetAnalysisResult();
			}
			else if (Status == TEXT("failed") || Status == TEXT("error"))
			{
				StopProgressTimer();
				FString ErrorMsg;
				if (JsonObject->HasField(TEXT("error_message")))
				{
					ErrorMsg = JsonObject->GetStringField(TEXT("error_message"));
				}
				else if (JsonObject->HasField(TEXT("error")))
				{
					ErrorMsg = JsonObject->GetStringField(TEXT("error"));
				}
				ShowErrorMessage(ErrorMsg);
				OnCancelAnalysisClicked();
			}
			
			// If server directly returned complete result
			if (JsonObject->HasField(TEXT("organ_stats")) || JsonObject->HasField(TEXT("mesh_data")))
			{
				UE_LOG(LogLiverAI, Warning, TEXT("Found complete result in progress response, processing..."));
				StopProgressTimer();
				bAnalysisInProgress = false;
				UpdateRunButtonState();
				UpdateProgress(100.0f, TEXT("Analysis complete!"));
				ShowAnalysisResults(ResponseContent);
			}
		}
	}
	else
	{
		UE_LOG(LogLiverAI, Warning, TEXT("Progress check failed, response code: %d"), 
			Response.IsValid() ? Response->GetResponseCode() : -1);
	}
}

// Get final analysis result
void SLiverAIWidget::GetAnalysisResult()
{
	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
	HttpRequest->OnProcessRequestComplete().BindSP(this, &SLiverAIWidget::OnResultResponse);
	HttpRequest->SetVerb("GET");
	HttpRequest->SetURL(ServerURL + TEXT("/api/result/") + CurrentRequestId);
	HttpRequest->SetTimeout(30.0f);

	if (!HttpRequest->ProcessRequest())
	{
		UpdateStatusText(TEXT("Error: Cannot get analysis result"));
		OnCancelAnalysisClicked();
	}
}

// Analysis result response handler
void SLiverAIWidget::OnResultResponse(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful)
{
	bAnalysisInProgress = false;
	UpdateRunButtonState();

	if (bWasSuccessful && Response.IsValid())
	{
		int32 ResponseCode = Response->GetResponseCode();
		if (ResponseCode == 200)
		{
			FString ResponseContent = Response->GetContentAsString();
			UpdateProgress(100.0f, TEXT("Analysis complete!"));
			ShowAnalysisResults(ResponseContent);
			UpdateStatusText(TEXT("AI analysis successfully completed!"));
		}
		else
		{
			FString ErrorMsg = FString::Printf(TEXT("Cannot get result (code: %d)"), ResponseCode);
			ShowErrorMessage(ErrorMsg);
			UpdateStatusText(ErrorMsg);
		}
	}
	else
	{
		ShowErrorMessage(TEXT("Network connection failed while getting result"));
		UpdateStatusText(TEXT("Failed to get analysis result"));
	}
}

// HTTP connection test response handler
void SLiverAIWidget::OnHttpRequestComplete(TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> Request, TSharedPtr<IHttpResponse, ESPMode::ThreadSafe> Response, bool bWasSuccessful)
{
	if (bWasSuccessful && Response.IsValid())
	{
		int32 ResponseCode = Response->GetResponseCode();
		if (ResponseCode == 200)
		{
			bServerHealthy = true;
			UpdateServerStatus(true);
			UpdateStatusText(FString::Printf(TEXT("Server connection successful! Response code: %d"), ResponseCode));
		}
		else
		{
			bServerHealthy = false;
			UpdateServerStatus(false);
			UpdateStatusText(FString::Printf(TEXT("Server response abnormal: %d"), ResponseCode));
		}
	}
	else
	{
		bServerHealthy = false;
		UpdateServerStatus(false);
		UpdateStatusText(TEXT("Server connection failed - Please check if server is running"));
	}

	UpdateRunButtonState();
}

void SLiverAIWidget::UpdateRunButtonState()
{
	FString MRIPath = MRIFilePathText->GetText().ToString();
	FString LiverPath = LiverModelPathText->GetText().ToString();
	FString VesselPath = VesselModelPathText->GetText().ToString();
	FString TumorPath = TumorModelPathText->GetText().ToString();
	
	bool bCanRun = bServerHealthy && 
				   !MRIPath.IsEmpty() && 
				   !LiverPath.IsEmpty() && 
				   !VesselPath.IsEmpty() &&
				   !TumorPath.IsEmpty() &&
				   !bAnalysisInProgress;
	
	if (RunAnalysisButton.IsValid())
	{
		RunAnalysisButton->SetEnabled(bCanRun);
	}
	
	if (CancelAnalysisButton.IsValid())
	{
		CancelAnalysisButton->SetEnabled(bAnalysisInProgress);
	}
}

void SLiverAIWidget::UpdateStatusText(const FString& Message)
{
	if (StatusText.IsValid())
	{
		StatusText->SetText(FText::FromString(Message));
	}
	
	UE_LOG(LogLiverAI, Log, TEXT("Status: %s"), *Message);
}

void SLiverAIWidget::UpdateServerStatus(bool bConnected)
{
	if (ServerStatusText.IsValid())
	{
		if (bConnected)
		{
			ServerStatusText->SetText(LOCTEXT("ServerConnected", "Server: Connected"));
			ServerStatusText->SetColorAndOpacity(FLinearColor::Green);
		}
		else
		{
			ServerStatusText->SetText(LOCTEXT("ServerDisconnected", "Server: Connection Failed"));
			ServerStatusText->SetColorAndOpacity(FLinearColor::Red);
		}
	}
}

void SLiverAIWidget::UpdateProgress(float Progress, const FString& Message)
{
	CurrentProgress = Progress;
	if (ProgressText.IsValid())
	{
		FString ProgressMessage = FString::Printf(TEXT("Progress: %.1f%% - %s"), Progress, *Message);
		ProgressText->SetText(FText::FromString(ProgressMessage));
		
		// Set color based on progress
		FLinearColor ProgressColor = FLinearColor::White;
		if (Progress >= 100.0f)
		{
			ProgressColor = FLinearColor::Green;
		}
		else if (Progress > 0.0f)
		{
			ProgressColor = FLinearColor::Yellow;
		}
		ProgressText->SetColorAndOpacity(ProgressColor);
	}
}

void SLiverAIWidget::StartProgressTimer()
{
	if (!CurrentRequestId.IsEmpty())
	{
		// Reset time counter
		LastProgressCheckTime = 0.0f;
		
		// Register active timer
		if (!ActiveTimerHandle.IsValid())
		{
			ActiveTimerHandle = RegisterActiveTimer(0.1f, 
				FWidgetActiveTimerDelegate::CreateSP(this, &SLiverAIWidget::OnProgressTimerTick));
		}
		
		UpdateStatusText(TEXT("Started monitoring analysis progress..."));
	}
}

void SLiverAIWidget::StopProgressTimer()
{
	// Cancel active timer
	if (ActiveTimerHandle.IsValid())
	{
		TSharedPtr<FActiveTimerHandle> PinnedHandle = ActiveTimerHandle.Pin();
		if (PinnedHandle.IsValid())
		{
			UnRegisterActiveTimer(PinnedHandle.ToSharedRef());
		}
		ActiveTimerHandle.Reset();
	}
}

EActiveTimerReturnType SLiverAIWidget::OnProgressTimerTick(double InCurrentTime, float InDeltaTime)
{
	// Accumulate time
	LastProgressCheckTime += InDeltaTime;
	
	// Check progress every 3 seconds
	if (LastProgressCheckTime >= 3.0f)
	{
		LastProgressCheckTime = 0.0f;
		
		if (bAnalysisInProgress && !CurrentRequestId.IsEmpty())
		{
			CheckAnalysisProgress();
			return EActiveTimerReturnType::Continue;  // Continue timer
		}
	}
	else if (bAnalysisInProgress)
	{
		return EActiveTimerReturnType::Continue;  // Continue timer
	}
	
	// Stop timer
	return EActiveTimerReturnType::Stop;
}

void SLiverAIWidget::CheckAnalysisProgress()
{
	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
	HttpRequest->OnProcessRequestComplete().BindSP(this, &SLiverAIWidget::OnProgressResponse);
	HttpRequest->SetVerb("GET");
	
	FString StatusURL = ServerURL + TEXT("/api/status/") + CurrentRequestId;
	HttpRequest->SetURL(StatusURL);
	HttpRequest->SetTimeout(10.0f);

	UE_LOG(LogLiverAI, Warning, TEXT("Checking progress at: %s"), *StatusURL);
	
	if (!HttpRequest->ProcessRequest())
	{
		UE_LOG(LogLiverAI, Error, TEXT("Failed to send progress check request"));
	}
}

void SLiverAIWidget::ShowAnalysisResults(const FString& Results)
{
    UE_LOG(LogLiverAI, Warning, TEXT("=== ShowAnalysisResults called ==="));
    
    FString DisplayText; // Declare DisplayText variable
    
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Results);
    
    if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
    {
        // Generate diagnostic report
        FString DiagnosticReport = GenerateDiagnosticReport(JsonObject);
        DisplayText = DiagnosticReport;
        
        // Update diagnostic text display
        if (DiagnosticResultsText.IsValid())
        {
            DiagnosticResultsText->SetText(FText::FromString(DiagnosticReport));
        }
        
        // Parse and display image data
        FString ImageDataBase64, SegmentationDataBase64;
        
        if (JsonObject->TryGetStringField(TEXT("image_data_base64"), ImageDataBase64) &&
            JsonObject->TryGetStringField(TEXT("segmentation_data_base64"), SegmentationDataBase64))
        {
            // Decode Base64 data
            TArray<uint8> ImageBytes, SegBytes;
            FBase64::Decode(ImageDataBase64, ImageBytes);
            FBase64::Decode(SegmentationDataBase64, SegBytes);
            
            // Get image dimensions
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
                // Convert byte data to float array
                TArray<float> ImageFloatData;
                ImageFloatData.SetNum(ImageBytes.Num() / sizeof(float));
                FMemory::Memcpy(ImageFloatData.GetData(), ImageBytes.GetData(), ImageBytes.Num());
                
                // Update slice viewers
                if (OriginalSliceViewer.IsValid())
                {
                    OriginalSliceViewer->SetImageData(
                        ImageFloatData,
                        ImageDimensions[0],
                        ImageDimensions[1],
                        ImageDimensions[2]
                    );
                }
                
                if (SegmentationSliceViewer.IsValid())
                {
                    SegmentationSliceViewer->SetImageData(
                        ImageFloatData,
                        ImageDimensions[0],
                        ImageDimensions[1],
                        ImageDimensions[2]
                    );
                    SegmentationSliceViewer->SetMaskData(
                        SegBytes,
                        ImageDimensions[0],
                        ImageDimensions[1],
                        ImageDimensions[2]
                    );
                }
                
                UE_LOG(LogLiverAI, Warning, TEXT("Updated slice viewers with image data"));
            }
        }
        
        // Check if 3D mesh data exists
        if (bGenerate3DMesh)
        {
            UE_LOG(LogLiverAI, Warning, TEXT("3D Mesh generation is enabled, checking for mesh_data..."));
            
            const TArray<TSharedPtr<FJsonValue>>* MeshDataArray = nullptr;
            bool bFoundMeshData = false;
            
            if (JsonObject->TryGetArrayField(TEXT("mesh_data"), MeshDataArray))
            {
                bFoundMeshData = true;
                UE_LOG(LogLiverAI, Warning, TEXT("Found mesh_data array with %d items"), MeshDataArray->Num());
            }
            else if (JsonObject->TryGetArrayField(TEXT("meshes"), MeshDataArray))
            {
                bFoundMeshData = true;
                UE_LOG(LogLiverAI, Warning, TEXT("Found meshes array with %d items"), MeshDataArray->Num());
            }
            else if (JsonObject->TryGetArrayField(TEXT("mesh"), MeshDataArray))
            {
                bFoundMeshData = true;
                UE_LOG(LogLiverAI, Warning, TEXT("Found mesh array with %d items"), MeshDataArray->Num());
            }
            
            if (bFoundMeshData && MeshDataArray && MeshDataArray->Num() > 0)
            {
                DisplayText += TEXT("\n\n📊 3D Mesh Generation Status:\n");
                DisplayText += FString::Printf(TEXT("✅ Found %d 3D model data, preparing visualization...\n"), MeshDataArray->Num());
                
                UE_LOG(LogLiverAI, Warning, TEXT("Calling Generate3DMeshFromResults..."));
                Generate3DMeshFromResults(JsonObject);
            }
            else
            {
                DisplayText += TEXT("\n\n⚠ 3D Mesh Generation: Server did not return mesh data\n");
                DisplayText += TEXT("Hint: Ensure Python backend returns mesh_data field\n");
                UE_LOG(LogLiverAI, Warning, TEXT("No mesh data found in server response!"));
                
                if (JsonObject->HasField(TEXT("organ_stats")))
                {
                    DisplayText += TEXT("\n🔧 Generating test cube for demonstration...\n");
                    GenerateTestCube();
                }
            }
        }
    }
    else
    {
        DisplayText = FString::Printf(TEXT("Raw result:\n%s"), *Results);
        UE_LOG(LogLiverAI, Error, TEXT("Failed to parse JSON response!"));
    }
    
    // Update status text
    UpdateStatusText(DisplayText);
}

// Continue in next file...
void SLiverAIWidget::GenerateTestCube()
{
	UE_LOG(LogLiverAI, Warning, TEXT("=== GenerateTestCube called ==="));
	
	// Get editor world
	UWorld* World = nullptr;
	if (GEditor)
	{
		World = GEditor->GetEditorWorldContext().World();
	}
	
	if (!World)
	{
		UE_LOG(LogLiverAI, Error, TEXT("Failed to get editor world!"));
		return;
	}
	
	// Create Actor
	FActorSpawnParameters SpawnParams;
	SpawnParams.Name = *FString::Printf(TEXT("TestCube_%s"), *CurrentRequestId);
	
	AActor* NewActor = World->SpawnActor<AActor>(AActor::StaticClass(), FVector(0, 0, 100), FRotator::ZeroRotator, SpawnParams);
	if (!NewActor)
	{
		UE_LOG(LogLiverAI, Error, TEXT("Failed to spawn actor!"));
		return;
	}
	
	// Create root component
	USceneComponent* RootComp = NewObject<USceneComponent>(NewActor, TEXT("RootComponent"));
	NewActor->SetRootComponent(RootComp);
	RootComp->RegisterComponent();
	
	// Create ProceduralMeshComponent
	UProceduralMeshComponent* MeshComp = NewObject<UProceduralMeshComponent>(NewActor, TEXT("TestCube"));
	MeshComp->AttachToComponent(RootComp, FAttachmentTransformRules::KeepRelativeTransform);
	MeshComp->RegisterComponent();
	
	// Define cube vertices
	TArray<FVector> Vertices = {
		FVector(-50, -50, -50), FVector(50, -50, -50), FVector(50, 50, -50), FVector(-50, 50, -50),
		FVector(-50, -50, 50), FVector(50, -50, 50), FVector(50, 50, 50), FVector(-50, 50, 50)
	};
	
	// Define triangles
	TArray<int32> Triangles = {
		0,1,2, 2,3,0,  // Bottom face
		4,7,6, 6,5,4,  // Top face
		0,4,5, 5,1,0,  // Front face
		2,6,7, 7,3,2,  // Back face
		0,3,7, 7,4,0,  // Left face
		1,5,6, 6,2,1   // Right face
	};
	
	// Create simple normals and UVs
	TArray<FVector> Normals;
	TArray<FVector2D> UV0;
	TArray<FLinearColor> VertexColors;
	TArray<FProcMeshTangent> Tangents;
	
	Normals.Init(FVector::UpVector, Vertices.Num());
	UV0.Init(FVector2D::ZeroVector, Vertices.Num());
	VertexColors.Init(FLinearColor(0.8f, 0.2f, 0.2f, 1.0f), Vertices.Num());
	
	// Create mesh
	MeshComp->CreateMeshSection_LinearColor(
		0, Vertices, Triangles, Normals, UV0, VertexColors, Tangents, true
	);
	
	// Set material
	UMaterial* DefaultMaterial = Cast<UMaterial>(StaticLoadObject(UMaterial::StaticClass(), nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")));
	
	if (DefaultMaterial)
	{
		MeshComp->SetMaterial(0, DefaultMaterial);
	}
	
	// Focus on new Actor
	if (GEditor)
	{
		GEditor->SelectNone(false, true);
		GEditor->SelectActor(NewActor, true, true);
		GEditor->NoteSelectionChange();
		GEditor->MoveViewportCamerasToActor(*NewActor, false);
	}
	
	UpdateStatusText(TEXT("✅ Test cube generated successfully!"));
	UE_LOG(LogLiverAI, Warning, TEXT("Test cube created successfully!"));
}

void SLiverAIWidget::OnGenerate3DMeshStateChanged(ECheckBoxState NewState)
{
	bGenerate3DMesh = (NewState == ECheckBoxState::Checked);
	UpdateStatusText(bGenerate3DMesh ? 
		TEXT("3D Mesh Generation: Enabled - 3D model will be automatically generated after analysis") : 
		TEXT("3D Mesh Generation: Disabled - Only analysis report will be displayed"));
}

void SLiverAIWidget::Generate3DMeshFromResults(TSharedPtr<FJsonObject> ResultsJson)
{
	if (!ResultsJson.IsValid())
	{
		UpdateStatusText(TEXT("Error: Invalid analysis result data"));
		return;
	}
	
	UpdateStatusText(TEXT("🎨 Generating 3D visualization model..."));
	
	// Get editor world
	UWorld* World = nullptr;
	if (GEditor)
	{
		World = GEditor->GetEditorWorldContext().World();
	}
	
	if (!World)
	{
		UpdateStatusText(TEXT("Error: Cannot get editor world"));
		return;
	}
	
	// Parse mesh_data
	const TArray<TSharedPtr<FJsonValue>>* MeshDataArray;
	if (!ResultsJson->TryGetArrayField(TEXT("mesh_data"), MeshDataArray))
	{
		UpdateStatusText(TEXT("Warning: No mesh_data in server returned data"));
		return;
	}
	
	// Prepare analysis result structure
	FLiverAnalysisResult AnalysisResult;
	AnalysisResult.RequestId = CurrentRequestId;
	AnalysisResult.Timestamp = FDateTime::Now().ToString();
	AnalysisResult.bSuccess = true;
	
	// Parse mesh data for each organ
	int32 MeshCount = 0;
	for (int32 i = 0; i < MeshDataArray->Num(); i++)
	{
		const TSharedPtr<FJsonValue>& MeshValue = (*MeshDataArray)[i];
		
		UE_LOG(LogLiverAI, Warning, TEXT("Processing mesh %d/%d"), i+1, MeshDataArray->Num());
		
		if (MeshValue->Type != EJson::Object)
		{
			UE_LOG(LogLiverAI, Error, TEXT("Mesh item %d is not an object"), i);
			continue;
		}
			
		TSharedPtr<FJsonObject> MeshObject = MeshValue->AsObject();
		
		// Create mesh data structure
		FMeshData MeshData;
		
		// Get organ name
		if (!MeshObject->TryGetStringField(TEXT("organ_name"), MeshData.OrganName))
		{
			UE_LOG(LogLiverAI, Error, TEXT("Mesh %d missing organ_name"), i);
			continue;
		}
		
		UE_LOG(LogLiverAI, Warning, TEXT("Processing organ: %s"), *MeshData.OrganName);
		
		// Parse vertex data
		const TArray<TSharedPtr<FJsonValue>>* VerticesArray;
		if (MeshObject->TryGetArrayField(TEXT("vertices"), VerticesArray))
		{
			UE_LOG(LogLiverAI, Warning, TEXT("Found vertices array with %d items"), VerticesArray->Num());
			
			for (int32 v = 0; v < VerticesArray->Num(); v++)
			{
				const TSharedPtr<FJsonValue>& VertexValue = (*VerticesArray)[v];
				const TArray<TSharedPtr<FJsonValue>>* VertexCoords = nullptr;
				
				if (VertexValue->TryGetArray(VertexCoords) && VertexCoords && VertexCoords->Num() == 3)
				{
					float X = (*VertexCoords)[0]->AsNumber();
					float Y = (*VertexCoords)[1]->AsNumber();
					float Z = (*VertexCoords)[2]->AsNumber();
					
					// UE uses centimeters, may need scaling
					MeshData.Vertices.Add(FVector(X * 10.0f, Y * 10.0f, Z * 10.0f));
				}
				else
				{
					UE_LOG(LogLiverAI, Error, TEXT("Invalid vertex format at index %d"), v);
				}
			}
		}
		else
		{
			UE_LOG(LogLiverAI, Error, TEXT("No vertices array found for %s"), *MeshData.OrganName);
		}
		
		// Parse triangle indices - try multiple possible field names
		const TArray<TSharedPtr<FJsonValue>>* TrianglesArray = nullptr;
		bool bFoundTriangles = false;
		
		if (MeshObject->TryGetArrayField(TEXT("triangles"), TrianglesArray))
		{
			bFoundTriangles = true;
		}
		else if (MeshObject->TryGetArrayField(TEXT("faces"), TrianglesArray))
		{
			bFoundTriangles = true;
		}
		else if (MeshObject->TryGetArrayField(TEXT("indices"), TrianglesArray))
		{
			bFoundTriangles = true;
		}
		else if (MeshObject->TryGetArrayField(TEXT("polygons"), TrianglesArray))
		{
			bFoundTriangles = true;
		}
		
		if (bFoundTriangles && TrianglesArray)
		{
			// Check if first element is a single integer or array
			if (TrianglesArray->Num() > 0)
			{
				const TSharedPtr<FJsonValue>& FirstElement = (*TrianglesArray)[0];
				
				// Check if nested array format
				const TArray<TSharedPtr<FJsonValue>>* FaceArray = nullptr;
				if (FirstElement->TryGetArray(FaceArray))
				{
					// Format: [[v1,v2,v3], [v4,v5,v6], ...]
					for (int32 f = 0; f < TrianglesArray->Num(); f++)
					{
						const TSharedPtr<FJsonValue>& FaceValue = (*TrianglesArray)[f];
						const TArray<TSharedPtr<FJsonValue>>* Face = nullptr;
						
						if (FaceValue->TryGetArray(Face) && Face && Face->Num() >= 3)
						{
							// Add three vertex indices of the triangle
							for (int32 v = 0; v < 3; v++)
							{
								int32 Index = (*Face)[v]->AsNumber();
								MeshData.Triangles.Add(Index);
							}
						}
					}
				}
				else
				{
					// Format: [v1,v2,v3,v4,v5,v6,...]
					for (int32 t = 0; t < TrianglesArray->Num(); t++)
					{
						MeshData.Triangles.Add((*TrianglesArray)[t]->AsNumber());
					}
				}
			}
		}
		
		// Set color (based on organ type)
		if (MeshData.OrganName.Contains(TEXT("Liver")))
		{
			MeshData.Color = FLinearColor(0.6f, 0.2f, 0.2f, 0.8f); // Dark red
		}
		else if (MeshData.OrganName.Contains(TEXT("Vessel")))
		{
			MeshData.Color = FLinearColor(0.2f, 0.2f, 0.8f, 0.8f); // Blue
		}
		else if (MeshData.OrganName.Contains(TEXT("Tumor")))
		{
			MeshData.Color = FLinearColor(0.8f, 0.8f, 0.2f, 0.9f); // Yellow
		}
		else
		{
			MeshData.Color = FLinearColor(0.5f, 0.5f, 0.5f, 0.8f); // Gray
		}
		
		// Add to result
		if (MeshData.Vertices.Num() > 0 && MeshData.Triangles.Num() > 0)
		{
			AnalysisResult.MeshData.Add(MeshData);
			
			// Add statistics
			FOrganVolumeStats Stats;
			Stats.OrganName = MeshData.OrganName;
			Stats.VoxelCount = MeshData.Vertices.Num();
			Stats.VolumeML = MeshObject->GetNumberField(TEXT("volume_ml"));
			Stats.NumComponents = 1;
			AnalysisResult.OrganStats.Add(Stats);
			
			MeshCount++;
		}
	}
	
	if (MeshCount == 0)
	{
		UpdateStatusText(TEXT("Warning: No valid mesh data to generate"));
		return;
	}
	
	// Generate Actor in world
	FActorSpawnParameters SpawnParams;
	SpawnParams.Name = *FString::Printf(TEXT("LiverAnalysis_%s"), *CurrentRequestId);
	
	AActor* NewActor = World->SpawnActor<AActor>(AActor::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);
	if (NewActor)
	{
		// Create root component
		USceneComponent* RootComp = NewObject<USceneComponent>(NewActor, TEXT("RootComponent"));
		NewActor->SetRootComponent(RootComp);
		RootComp->RegisterComponent();
		
		// Create ProceduralMeshComponent for each organ
		for (const FMeshData& MeshData : AnalysisResult.MeshData)
		{
			FString CompName = FString::Printf(TEXT("%s_Mesh"), *MeshData.OrganName);
			UProceduralMeshComponent* MeshComp = NewObject<UProceduralMeshComponent>(NewActor, *CompName);
			
			if (MeshComp)
			{
				MeshComp->AttachToComponent(RootComp, FAttachmentTransformRules::KeepRelativeTransform);
				MeshComp->RegisterComponent();
				
				// Generate normals and tangents
				TArray<FVector> Normals;
				TArray<FVector2D> UV0;
				TArray<FLinearColor> VertexColors;
				TArray<FProcMeshTangent> Tangents;
				
				// Simple normal calculation
				Normals.Init(FVector::UpVector, MeshData.Vertices.Num());
				UV0.Init(FVector2D::ZeroVector, MeshData.Vertices.Num());
				VertexColors.Init(MeshData.Color, MeshData.Vertices.Num());
				
				// Create mesh section
				MeshComp->CreateMeshSection_LinearColor(
					0,
					MeshData.Vertices,
					MeshData.Triangles,
					Normals,
					UV0,
					VertexColors,
					Tangents,
					true
				);
				
				// Set material
				UMaterial* DefaultMaterial = Cast<UMaterial>(StaticLoadObject(UMaterial::StaticClass(), nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial")));
				
				if (DefaultMaterial)
				{
					UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(DefaultMaterial, MeshComp);
					if (DynMaterial)
					{
						DynMaterial->SetVectorParameterValue(TEXT("Color"), MeshData.Color);
						MeshComp->SetMaterial(0, DynMaterial);
					}
				}
				
				MeshComp->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
			}
		}
		
		// Move Actor near viewport center
		NewActor->SetActorLocation(FVector(0, 0, 100));
		
		// Focus on newly created Actor
		if (GEditor)
		{
			GEditor->SelectNone(false, true);
			GEditor->SelectActor(NewActor, true, true);
			GEditor->NoteSelectionChange();
			GEditor->MoveViewportCamerasToActor(*NewActor, false);
		}
		
		UpdateStatusText(FString::Printf(TEXT("✅ Successfully generated 3D model! Contains %d organ meshes"), MeshCount));
		UE_LOG(LogLiverAI, Log, TEXT("Successfully created 3D visualization with %d organ meshes"), MeshCount);
	}
	else
	{
		UpdateStatusText(TEXT("Error: Cannot create Actor in scene"));
	}
}

void SLiverAIWidget::ShowErrorMessage(const FString& ErrorMsg)
{
	FString ErrorText = FString::Printf(TEXT("❌ Analysis failed!\n\nError message: %s\n\nPlease check:\n• Server connection status\n• File paths are correct\n• Model files are valid\n• Server log information"), *ErrorMsg);
	
	if (StatusText.IsValid())
	{
		StatusText->SetText(FText::FromString(ErrorText));
		StatusText->SetColorAndOpacity(FLinearColor::Red);
	}
	
	UpdateProgress(0.0f, TEXT("Analysis failed"));
}



FString SLiverAIWidget::GenerateDiagnosticReport(TSharedPtr<FJsonObject> ResultsJson)
{
    FString Report;
    Report += TEXT("=====================================\n");
    Report += TEXT("   AI Liver Analysis - Diagnostic Report\n");
    Report += TEXT("=====================================\n\n");
    Report += FString::Printf(TEXT("Analysis Time: %s\n"), *FDateTime::Now().ToString());
    Report += FString::Printf(TEXT("Request ID: %s\n\n"), *CurrentRequestId);
    
    // Parse organ statistics
    const TArray<TSharedPtr<FJsonValue>>* OrganStats;
    if (ResultsJson->TryGetArrayField(TEXT("organ_stats"), OrganStats))
    {
        Report += TEXT("【Organ Segmentation Results】\n");
        Report += TEXT("─────────────────────────────\n\n");
        
        float TotalLiverVolume = 0.0f;
        float TotalTumorVolume = 0.0f;
        int32 TumorCount = 0;
        
        for (const auto& StatValue : *OrganStats)
        {
            TSharedPtr<FJsonObject> StatObject = StatValue->AsObject();
            FString OrganName = StatObject->GetStringField(TEXT("organ_name"));
            float Volume = StatObject->GetNumberField(TEXT("volume_ml"));
            int32 VoxelCount = StatObject->GetIntegerField(TEXT("voxel_count"));
            int32 NumComponents = StatObject->HasField(TEXT("num_components")) ? 
                StatObject->GetIntegerField(TEXT("num_components")) : 1;
            
            // Convert organ names
            FString DisplayName = OrganName;
            if (OrganName.Contains(TEXT("liver")))
            {
                DisplayName = TEXT("Liver");
                TotalLiverVolume = Volume;
            }
            else if (OrganName.Contains(TEXT("vessel")))
            {
                DisplayName = TEXT("Hepatic Vessels");
            }
            else if (OrganName.Contains(TEXT("tumor")))
            {
                DisplayName = TEXT("Tumor");
                TotalTumorVolume = Volume;
                TumorCount = NumComponents;
            }
            
            Report += FString::Printf(TEXT("● %s:\n"), *DisplayName);
            Report += FString::Printf(TEXT("  Volume: %.2f mL\n"), Volume);
            Report += FString::Printf(TEXT("  Voxel Count: %d\n"), VoxelCount);
            if (NumComponents > 1)
            {
                Report += FString::Printf(TEXT("  Connected Components: %d\n"), NumComponents);
            }
            Report += TEXT("\n");
        }
        
        // Clinical assessment
        Report += TEXT("\n【Clinical Assessment】\n");
        Report += TEXT("─────────────────────────────\n\n");
        
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
                Report += TEXT("  → Clinical correlation and follow-up recommended\n");
            }
            else
            {
                Report += FString::Printf(TEXT("⚠ Multiple lesions detected (%d lesions, total volume: %.1f mL)\n"), 
                    TumorCount, TotalTumorVolume);
                Report += TEXT("  → Further evaluation to rule out metastatic disease recommended\n");
            }
        }
        else
        {
            Report += TEXT("✓ No significant hepatic lesions detected\n");
        }
    }
    
    // Add diagnostic report field (if exists)
    FString DiagnosticReport;
    if (ResultsJson->TryGetStringField(TEXT("diagnostic_report"), DiagnosticReport) && !DiagnosticReport.IsEmpty())
    {
        Report += TEXT("\n【Detailed Diagnosis】\n");
        Report += TEXT("─────────────────────────────\n");
        Report += DiagnosticReport + TEXT("\n");
    }
    
    // Technical notes
    Report += TEXT("\n【Technical Notes】\n");
    Report += TEXT("─────────────────────────────\n");
    Report += TEXT("• Analysis performed using deep learning AI models\n");
    Report += TEXT("• 3D reconstruction using Marching Cubes algorithm\n");
    Report += TEXT("• Results for research/educational purposes only\n");
    Report += TEXT("• Cannot replace professional medical diagnosis\n");
    Report += TEXT("=====================================\n");
    
    return Report;
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FLiverImageAIModule, LiverImageAI)
