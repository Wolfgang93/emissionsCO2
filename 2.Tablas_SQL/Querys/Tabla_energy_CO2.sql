-- Active: 1662947452541@@localhost@3306@co2
USE co2;
DROP TABLE IF EXISTS`energy_CO2_by_country`;                      
CREATE TABLE IF NOT EXISTS `energy_CO2_by_country`(
	`country` VARCHAR(30),
    `energy_type` VARCHAR(30),
    `year` INT,
	`energy_consumption` FLOAT,
    `energy_production` FLOAT,
    `gdp` VARCHAR(255),
    `population` INT,
    `intensityPerCapita` FLOAT,
    `intensityByGDP` FLOAT,
    `co2_emission` DECIMAL (10,3));
LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\CO2\\energyco2_by_country.csv'
INTO TABLE `energy_CO2_by_country`
FIELDS TERMINATED BY ',' ENCLOSED BY '' ESCAPED BY '' 
LINES TERMINATED BY '\n' IGNORE 1 LINES;

DROP TABLE IF EXISTS`energy_CO2_world`;                      
CREATE TABLE IF NOT EXISTS `energy_CO2_world`(
	`world` VARCHAR(30),
    `energy_type` VARCHAR(30),
    `year` INT,
	`energy_consumption` FLOAT,
    `energy_production` FLOAT,
    `gdp` FLOAT,
    `population` INT,
    `Energy_intensity_per_capita` FLOAT,
    `Energy_intensity_by_GDP` FLOAT,
    `co2_emission` DECIMAL (10,3));
LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\CO2\\energyco2_worlds.csv'
INTO TABLE `energy_CO2_world`
FIELDS TERMINATED BY ',' ENCLOSED BY '' ESCAPED BY '' 
LINES TERMINATED BY '\n' IGNORE 1 LINES;