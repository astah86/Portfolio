
CREATE DATABASE /*!32312 IF NOT EXISTS*/`monitoring` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `monitoring`;

CREATE USER 'grafana'@'%' IDENTIFIED BY '123456';
GRANT SELECT ON `monitoring`.* TO 'grafana'@'%';
ALTER USER 'root'@'%' IDENTIFIED BY 'p76Se7BoVbrn';
FLUSH PRIVILEGES;

FLUSH PRIVILEGES;

DROP TABLE IF EXISTS `metrics`;

CREATE TABLE `metrics` (
    `timestamp` datetime NOT NULL,
    `cpu_usage` decimal(5, 2) NOT NULL,
    `mem_available` bigint NOT NULL,
    `reqs_per_min` int(11) NOT NULL,
    `time_of_proc` decimal(5, 2) NOT NULL,
    PRIMARY KEY (timestamp)
) ENGINE=InnoDB;
    
INSERT  INTO `metrics`(`timestamp`,`cpu_usage`, `mem_available`, `reqs_per_min`, `time_of_proc`) VALUES
