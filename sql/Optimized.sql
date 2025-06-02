CREATE FUNCTION [dbo].[PredictiveInputDataOptimized]
(
    @ufmID INT
)
RETURNS @outputTable TABLE
(
    UserForecastMethodID int,
    PodID varchar(15),
    CustomerID varchar(10),
    TariffID varchar(500),
    ReportingMonth date,
    PeakConsumption decimal(14,2),
    StandardConsumption decimal(14,2),
    OffPeakConsumption decimal(14,2),
    Block1Consumption decimal(14,2),
    Block2Consumption decimal(14,2),
    Block3Consumption decimal(14,2),
    Block4Consumption decimal(14,2),
    NonTOUConsumption decimal(14,2)
)
AS
BEGIN
    DECLARE
        @json NVARCHAR(MAX),
        @dateJson NVARCHAR(MAX),
        @monthTable TABLE ([Month] DATE),
        @yearTable TABLE ([Year] INT),
        @filterTable TABLE (
            PodID VARCHAR(15),
            CustomerID VARCHAR(10),
            Cluster VARCHAR(50),
            OperatingUnit VARCHAR(50),
            CSA VARCHAR(50)
        );

    SELECT
        @json = JSONCustomer,
        @dateJson = DateJSON
    FROM dbo.UserForecastMethod
    WHERE UserForecastMethodID = @ufmID;

    -- Filter JSONs into values
    DECLARE @month NVARCHAR(MAX), @year NVARCHAR(MAX),
            @pods NVARCHAR(MAX), @cust NVARCHAR(MAX),
            @clus NVARCHAR(MAX), @csas NVARCHAR(MAX), @ous NVARCHAR(MAX);

    SELECT
        @month = [MONTH],
        @year = [YEAR]
    FROM OPENJSON(@dateJson)
    WITH ([MONTH] NVARCHAR(MAX) '$.Month', [YEAR] NVARCHAR(MAX) '$.Year');

    SELECT
        @pods = PodId,
        @cust = CustomerID,
        @clus = Cluster,
        @csas = CSA,
        @ous = OU
    FROM OPENJSON(@json)
    WITH (
        PodId NVARCHAR(MAX) '$.PodId',
        CustomerID NVARCHAR(MAX) '$.CustomerID',
        Cluster NVARCHAR(MAX) '$.Cluster',
        CSA NVARCHAR(MAX) '$.CSA',
        OU NVARCHAR(MAX) '$.OU'
    );

    -- Build filtered dimension table (single pass)
    INSERT INTO @filterTable
    SELECT
        dp.PodID, dp.CustomerID, do.Cluster, do.OperatingUnit, do.CustomerServiceAreaDescription
    FROM dbo.DimPOD dp
    LEFT JOIN dbo.DimOperatingUnit do ON do.OperatingUnitID = dp.OperatingUnit
    WHERE
        (@pods IS NULL OR dp.PodID IN (SELECT value FROM OPENJSON(@pods))) AND
        (@cust IS NULL OR dp.CustomerID IN (SELECT value FROM OPENJSON(@cust))) AND
        (@clus IS NULL OR do.Cluster IN (SELECT value FROM OPENJSON(@clus))) AND
        (@ous IS NULL OR do.OperatingUnit IN (SELECT value FROM OPENJSON(@ous))) AND
        (@csas IS NULL OR do.CustomerServiceAreaDescription IN (SELECT value FROM OPENJSON(@csas)));

    -- Filter time table (once)
    IF @month IS NOT NULL
        INSERT INTO @monthTable
        SELECT CONVERT(DATE, value) FROM OPENJSON(@month);

    IF @year IS NOT NULL
        INSERT INTO @yearTable
        SELECT CONVERT(INT, value) FROM OPENJSON(@year);

    -- Insert final data
    INSERT INTO @outputTable
    SELECT
        @ufmID,
        a.PodID,
        p.CustomerID,
        p.TariffID,
        a.ReportingMonth,
        SUM(a.PeakConsumption),
        SUM(a.StandardConsumption),
        SUM(a.OffPeakConsumption),
        SUM(a.Block1Consumption),
        SUM(a.Block2Consumption),
        SUM(a.Block3Consumption),
        SUM(a.Block4Consumption),
        SUM(a.NonTOUConsumption)
    FROM dbo.ActualData a
    INNER JOIN dbo.DimPOD p ON a.PodID = p.PodID
    INNER JOIN @filterTable f ON f.PodID = p.PodID
    WHERE
        a.PeakConsumption > 0 AND
        a.StandardConsumption > 0 AND
        a.OffPeakConsumption > 0 AND
        a.ReportingMonth IN (SELECT [Month] FROM @monthTable)
    GROUP BY a.PodID, p.CustomerID, p.TariffID, a.ReportingMonth;

    RETURN;
END;
GO
