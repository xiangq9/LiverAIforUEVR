using UnrealBuildTool;

public class LiverImageAI : ModuleRules
{
    public LiverImageAI(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] 
        {
            "Core",
            "CoreUObject",
            "Engine",
            "Slate",
            "SlateCore",
            "HTTP",
            "Json",
            "JsonUtilities",
            "ProceduralMeshComponent",
            "RenderCore",
            "RHI"
        });

        PrivateDependencyModuleNames.AddRange(new string[] 
        {
            "UnrealEd",
            "ToolMenus",
            "EditorSubsystem",
            "AppFramework",
            "DesktopPlatform",
            "EditorStyle",
            "ToolWidgets",
            "Projects",
            "EditorWidgets"
        });

        // Explicitly specify source files for the modular structure
        PublicIncludePaths.AddRange(new string[] 
        {
            "LiverImageAI/Public"
        });

        PrivateIncludePaths.AddRange(new string[] 
        {
            "LiverImageAI/Private"
        });
    }
}
