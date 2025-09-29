#include "Services/FileService.h"
#include "HAL/PlatformFilemanager.h"
#include "IDesktopPlatform.h"
#include "DesktopPlatformModule.h"
#include "Framework/Application/SlateApplication.h"
#include "Misc/Paths.h"

FFileService::FFileService()
{
}

FFileService::~FFileService()
{
}

bool FFileService::ValidateFilePath(const FString& FilePath) const
{
    return !FilePath.IsEmpty() && FileExists(FilePath);
}

bool FFileService::FileExists(const FString& FilePath) const
{
    return FPlatformFileManager::Get().GetPlatformFile().FileExists(*FilePath);
}

FString FFileService::GetFileExtension(const FString& FilePath) const
{
    return FPaths::GetExtension(FilePath);
}

int64 FFileService::GetFileSize(const FString& FilePath) const
{
    return FPlatformFileManager::Get().GetPlatformFile().FileSize(*FilePath);
}

FString FFileService::FormatFileSize(int64 SizeInBytes) const
{
    if (SizeInBytes < 1024)
        return FString::Printf(TEXT("%lld B"), SizeInBytes);
    else if (SizeInBytes < 1024 * 1024)
        return FString::Printf(TEXT("%.1f KB"), SizeInBytes / 1024.0);
    else if (SizeInBytes < 1024 * 1024 * 1024)
        return FString::Printf(TEXT("%.1f MB"), SizeInBytes / (1024.0 * 1024.0));
    else
        return FString::Printf(TEXT("%.1f GB"), SizeInBytes / (1024.0 * 1024.0 * 1024.0));
}

bool FFileService::OpenFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename)
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> OpenFilenames;
        bool bOpened = DesktopPlatform->OpenFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
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

bool FFileService::SaveFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename)
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> SaveFilenames;
        bool bSaved = DesktopPlatform->SaveFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            Title,
            TEXT(""),
            TEXT(""),
            FileTypes,
            EFileDialogFlags::None,
            SaveFilenames
        );

        if (bSaved && SaveFilenames.Num() > 0)
        {
            OutFilename = SaveFilenames[0];
            return true;
        }
    }
    return false;
}