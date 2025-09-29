#pragma once
#include "CoreMinimal.h"

class LIVERIMAGEAI_API FImageDataProcessor
{
public:
    static TArray<float> ConvertBase64ToFloatArray(const FString& Base64String);
    static TArray<uint8> ConvertBase64ToByteArray(const FString& Base64String);
    static void NormalizeImageData(TArray<float>& ImageData);
    static FLinearColor GetOrganColor(const FString& OrganName);
    static FString GenerateRequestId();
};