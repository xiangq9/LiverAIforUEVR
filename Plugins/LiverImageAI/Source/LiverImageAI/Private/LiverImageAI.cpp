#include "LiverImageAI.h"
#include "UI/MainAnalysisWidget.h"
#include "Types/LiverAITypes.h"
#include "ToolMenus.h"
#include "Framework/Docking/TabManager.h"
#include "Widgets/Docking/SDockTab.h"

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

    UToolMenus::RegisterStartupCallback(
        FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FLiverImageAIModule::RegisterMenus));
    
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
            SNew(SMainAnalysisWidget)
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
    UI_COMMAND(OpenPluginWindow, "Liver AI", "Open Liver AI Analysis window", 
        EUserInterfaceActionType::Button, FInputChord());
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FLiverImageAIModule, LiverImageAI)