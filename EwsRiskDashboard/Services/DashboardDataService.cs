using System.Net.Http.Json;
using System.Text.Json;
using EwsRiskDashboard.Models;

namespace EwsRiskDashboard.Services;

// Loads dashboard_data.json (currently a static file under wwwroot/data) —
// swap the fetch below for a real API call once one exists, nothing else
// in the pages should need to change since they only depend on DashboardData.
public class DashboardDataService(HttpClient http)
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private DashboardData? _cache;

    public async Task<DashboardData> GetDashboardDataAsync()
    {
        _cache ??= await http.GetFromJsonAsync<DashboardData>("data/dashboard_data.json", JsonOptions)
            ?? throw new InvalidOperationException("dashboard_data.json returned no data.");
        return _cache;
    }
}
