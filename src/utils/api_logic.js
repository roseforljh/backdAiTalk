/**
 * Core business logic for handling API requests, shared across handlers.
 * This mirrors the logic from the backend-docker version.
 */

/**
 * Checks if the provided API address belongs to an official Google domain.
 * @param {string} apiAddress The API address to check.
 * @returns {boolean} True if it's a Google domain, false otherwise.
 */
export function isGoogleOfficialAPI(apiAddress) {
    if (!apiAddress) {
        return false;
    }
    try {
        const url = new URL(apiAddress);
        const domain = url.hostname.toLowerCase();
        const googleDomains = [
            'generativelanguage.googleapis.com',
            'aiplatform.googleapis.com',
            'googleapis.com',
            'ai.google.dev'
        ];
        return googleDomains.some(googleDomain =>
            domain === googleDomain || domain.endsWith('.' + googleDomain)
        );
    } catch (error) {
        console.warn(`Failed to parse API address '${apiAddress}':`, error);
        return false;
    }
}

/**
 * Parses the incoming request body, supporting both JSON and multipart/form-data.
 * This is crucial for compatibility with the Android client.
 * @param {Request} request The incoming request object.
 * @returns {Promise<object>} The parsed request data.
 */
export async function parseRequestBody(request) {
    const contentType = request.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
        return await request.json();
    } else if (contentType.includes('multipart/form-data')) {
        const formData = await request.formData();
        const chatRequestJson = formData.get('chat_request_json');
        if (!chatRequestJson) {
            throw new Error('Bad Request: Missing "chat_request_json" field in form data.');
        }
        const requestData = JSON.parse(chatRequestJson);

        // Attach uploaded files to the request data object
        const uploadedFiles = [];
        for (const [key, value] of formData.entries()) {
            if (key !== 'chat_request_json' && value instanceof File) {
                uploadedFiles.push(value);
            }
        }
        if (uploadedFiles.length > 0) {
            requestData.uploadedFiles = uploadedFiles;
        }
        return requestData;
    } else {
        throw new Error(`Unsupported Content-Type: ${contentType}`);
    }
}

/**
 * Generates a unique request ID.
 * @returns {string} A UUID.
 */
export function generateRequestId() {
    return crypto.randomUUID();
}

/**
 * Converts an ArrayBuffer to a Base64 encoded string.
 * @param {ArrayBuffer} buffer The buffer to convert.
 * @returns {string} The Base64 string.
 */
export function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}