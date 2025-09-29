#pragma once
#include "CoreMinimal.h"

class LIVERIMAGEAI_API IFileService
{
public:
    virtual ~IFileService() = default;
    
    virtual bool ValidateFilePath(const FString& FilePath) const = 0;
    virtual bool FileExists(const FString& FilePath) const = 0;
    virtual FString GetFileExtension(const FString& FilePath) const = 0;
    virtual int64 GetFileSize(const FString& FilePath) const = 0;
    virtual FString FormatFileSize(int64 SizeInBytes) const = 0;
    virtual bool OpenFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename) = 0;
    virtual bool SaveFileDialog(const FString& Title, const FString& FileTypes, FString& OutFilename) = 0;
};