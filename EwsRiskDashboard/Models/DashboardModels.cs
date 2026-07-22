namespace EwsRiskDashboard.Models;

// Mirrors dashboard_data.json, produced by risk_scoring_poc.py.
// Field names match the JSON exactly (camelCase) — see Services/DashboardDataService.cs
// for the JsonSerializerOptions that binds them onto these PascalCase properties.

public record DashboardData
{
    public required ConvenorData Convenor { get; init; }
    public required AdvisorData Advisor { get; init; }
}

public record ConvenorData
{
    public required string Module { get; init; }
    public required List<ConvenorStudent> Students { get; init; }
}

public record AdvisorData
{
    public required List<AdvisorStudent> Students { get; init; }
}

public record RiskFactor
{
    public required string Label { get; init; }
    public required string Direction { get; init; } // "increases" or "decreases" risk
    public double? Value { get; init; } // raw feature value at scoring time (convenor only)
}

public record ConvenorStudent
{
    public required string Id { get; init; }
    public double Attendance { get; init; }
    public double Submission { get; init; }
    public double Formative { get; init; }
    public double Risk { get; init; }
    public required string Tier { get; init; } // critical | high | medium | low
    public required List<RiskFactor> Factors { get; init; }
}

public record ModuleRisk
{
    public required string Module { get; init; }
    public double Attendance { get; init; }
    public double Submission { get; init; }
    public double Formative { get; init; }
    public double Risk { get; init; }
    public required string Tier { get; init; }
    public required List<RiskFactor> Factors { get; init; }
}

public record AdvisorStudent
{
    public required string Id { get; init; }
    public required string Programme { get; init; }
    public int Quintile { get; init; }
    public bool Nsfas { get; init; }
    public bool FundingDisruption { get; init; }
    public int YearOfStudy { get; init; }
    public int MinYearsToComplete { get; init; }
    public double Risk { get; init; }
    public required string Tier { get; init; } // year-failure risk tier
    public required List<RiskFactor> Factors { get; init; }
    public double DelayedGradRisk { get; init; }
    public required string DelayedGradTier { get; init; }
    public required List<RiskFactor> DelayedGradFactors { get; init; }

    // Every module this student is enrolled in during their current academic
    // year, each scored independently — this is what lets the advisor drawer
    // show *which* module is driving the aggregate year-risk score.
    public required List<ModuleRisk> Modules { get; init; }

    // Computed pace fact, not a model output (see risk_scoring_poc.py Model C
    // docstring for why year-of-study can't honestly be a delayed-graduation
    // model *input* — it's still a real, honest fact worth surfacing).
    public bool OverMinimumDuration => YearOfStudy > MinYearsToComplete;
}
