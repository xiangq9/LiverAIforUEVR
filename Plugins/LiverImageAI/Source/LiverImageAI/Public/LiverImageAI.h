#pragma once
#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Framework/Commands/Commands.h"

class FToolBarBuilder;
class FMenuBuilder;

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

class FLiverImageAIModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
    
private:
    void PluginButtonClicked();
    void RegisterMenus();
    TSharedRef<class SDockTab> OnSpawnPluginTab(const class FSpawnTabArgs& SpawnTabArgs);
    
    TSharedPtr<class FUICommandList> PluginCommands;
};