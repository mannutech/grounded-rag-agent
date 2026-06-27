# Rate limits

Each API key is limited to **100 requests per second**, with a short burst
allowance of up to **200 requests per second**. When the limit is exceeded the
API returns HTTP **429** with a `Retry-After` header indicating how many seconds
to wait before retrying.

Rate limits are not fixed forever: they **can be raised on request** for
production workloads by contacting support with your expected throughput.
