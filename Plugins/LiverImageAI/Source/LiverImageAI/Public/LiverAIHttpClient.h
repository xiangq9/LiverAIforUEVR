#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "LiverAITypes.h"
#include "LiverAIHttpClient.generated.h"

// Event delegate declarations
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnServerHealthCheck, bool, bHealthy, const FString&, StatusMessage);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAnalysisProgress, const FAnalysisProgress&, Progress);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAnalysisComplete, const FLiverAnalysisResult&, Result);

UCLASS(BlueprintType)
class LIVERIMAGEAI_API ULiverAIHttpClient : public UObject
{
	GENERATED_BODY()

public:
	ULiverAIHttpClient();

	// Event delegates
	UPROPERTY(BlueprintAssignable, Category = "Events")
	FOnServerHealthCheck OnServerHealthCheck;

	UPROPERTY(BlueprintAssignable, Category = "Events")
	FOnAnalysisProgress OnAnalysisProgress;

	UPROPERTY(BlueprintAssignable, Category = "Events")
	FOnAnalysisComplete OnAnalysisComplete;

	// HTTP request functions
	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	void SetServerURL(const FString& URL);

	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	void CheckServerHealth();

	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	void SendAnalysisRequest(const FLiverAnalysisRequest& Request);

	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	void GetAnalysisProgress(const FString& RequestId);

	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	void GetAnalysisResult(const FString& RequestId);

	// Basic status queries
	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	bool IsServerHealthy() const;

	UFUNCTION(BlueprintCallable, Category = "Liver AI")
	FString GetServerURL() const;

private:
	// HTTP response handlers
	void OnAnalysisRequestResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully);
	void OnHealthCheckResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully);
	void OnProgressResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully);
	void OnResultResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully);

	// JSON parsing helper functions
	TSharedPtr<FJsonObject> ParseJsonResponse(FHttpResponsePtr Response);
	FLiverAnalysisResult ParseAnalysisResult(TSharedPtr<FJsonObject> JsonObject);
	FAnalysisProgress ParseAnalysisProgress(TSharedPtr<FJsonObject> JsonObject);
	TArray<FOrganVolumeStats> ParseOrganStats(const TArray<TSharedPtr<FJsonValue>>& JsonArray);

	// Configuration
	UPROPERTY()
	FString PythonServerURL;

	UPROPERTY()
	float RequestTimeout;

	UPROPERTY()
	bool bServerHealthy;
};

// Enhanced utility class
UCLASS(BlueprintType)
class LIVERIMAGEAI_API ULiverAIUtilities : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "Liver AI Utils")
	static FString GenerateRequestId();

	UFUNCTION(BlueprintCallable, Category = "Liver AI Utils")
	static FLinearColor GetOrganColor(const FString& OrganName);

	UFUNCTION(BlueprintCallable, Category = "Liver AI Utils")
	static FString GetCurrentTimeString();

	UFUNCTION(BlueprintCallable, Category = "Liver AI Utils")
	static bool ValidateFilePath(const FString& FilePath);

	UFUNCTION(BlueprintCallable, Category = "Liver AI Utils")
	static FString FormatFileSize(int64 SizeInBytes);

	UFUNCTION(BlueprintCallable, Category = "Liver AI Utils")
	static FString GetFileExtension(const FString& FilePath);
};