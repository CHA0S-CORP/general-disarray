import type {
	IDataObject,
	IExecuteFunctions,
	IHookFunctions,
	IHttpRequestMethods,
	ILoadOptionsFunctions,
	JsonObject,
} from 'n8n-workflow';
import { NodeApiError } from 'n8n-workflow';

/**
 * Make a request to the SIP agent REST API using the sipAgentApi credential.
 * Injects the API token as X-API-Key or Authorization: Bearer per the credential's
 * headerStyle, only when a token is set (auth is optional on the agent).
 */
export async function sipAgentApiRequest(
	this: IExecuteFunctions | ILoadOptionsFunctions | IHookFunctions,
	method: IHttpRequestMethods,
	endpoint: string,
	body?: IDataObject,
	qs?: IDataObject,
): Promise<IDataObject> {
	const credentials = await this.getCredentials('sipAgentApi');
	const headers: IDataObject = {};
	const token = (credentials.apiToken as string) || '';
	if (token) {
		if (credentials.headerStyle === 'bearer') {
			headers.Authorization = `Bearer ${token}`;
		} else {
			headers['X-API-Key'] = token;
		}
	}
	const baseUrl = (credentials.baseUrl as string).replace(/\/+$/, '');
	try {
		return (await this.helpers.httpRequest({
			method,
			url: `${baseUrl}${endpoint}`,
			headers,
			body,
			qs,
			json: true,
		})) as IDataObject;
	} catch (error) {
		// Surface the agent's FastAPI error detail as the headline message instead
		// of n8n's generic HTTP-status text (e.g. "No active call to speak to"
		// rather than "The resource you are requesting could not be found").
		const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data
			?.detail;
		let message: string | undefined;
		if (typeof detail === 'string' && detail) {
			message = detail;
		} else if (Array.isArray(detail)) {
			// FastAPI 422 validation errors: [{loc, msg, type}, ...]
			message = detail
				.map((d) => (d && typeof d === 'object' && 'msg' in d ? String(d.msg) : JSON.stringify(d)))
				.join('; ');
		}
		throw new NodeApiError(this.getNode(), error as JsonObject, message ? { message } : undefined);
	}
}
