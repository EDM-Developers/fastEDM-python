import numpy as np
from prettytable import PrettyTable

def reduce_dataframe_rows(df, row_limit):
    df = df.copy()
    while df.shape[0] > row_limit:
        df = df.iloc[::2, :]
    return df

def dataframe_to_prettytable(df):
    pt = PrettyTable()
    pt.field_names = df.columns
    for _, row in df.iterrows():
        E, library, theta, rho, mae = row
        pt.add_row((int(E), int(library), round(theta, 5), round(rho, 5), round(mae, 5)))
    return pt

class EasyEDMSummary:
    ROW_LIMIT = 10
    WIDTH_LIMIT = 80

    embedding_info = None
    theta_info = None
    nonlin_info = None
    lag_info = None
    conv_info = None

    def captureEmbeddingInfo(self, data):
        self.embedding_info = data

    def captureThetaInfo(self, data):
        self.theta_info = data

    def captureNonLinearTestInfo(self, data):
        self.nonlin_info = data

    def captureLagInfo(self, data):
        self.lag_info = data

    def captureConvergenceInfo(self, data):
        self.conv_info = data

    def printSummary(self):
        self.printEmbeddingInfo()
        self.printThetaInfo()
        self.printNonLinearTestInfo()
        self.printLagInfo()
        self.printConvergenceInfo()

    def printTable(self, table, message):
        # Center-align heading message
        paddedMessage = (" " * 10) + message

        widthPerCell = len(paddedMessage) // len(table.field_names)
        table._min_width = {k: widthPerCell for k in table.field_names}

        # Get final width of table to format header section
        tableAsStr = str(table)
        index = 0
        while tableAsStr[index] != "\n" and index < len(tableAsStr):
            index += 1

        print()
        print("-" * index)
        print(paddedMessage)
        print(table)

    def printEmbeddingInfo(self):
        if self.embedding_info is None:
            return
        df = reduce_dataframe_rows(self.embedding_info, self.ROW_LIMIT)
        pt = dataframe_to_prettytable(df)
        self.printTable(pt, "Finding optimal E using simplex projection")

    def printThetaInfo(self):
        if self.theta_info is None:
            return
        
        df = reduce_dataframe_rows(self.theta_info, self.ROW_LIMIT)
        pt = dataframe_to_prettytable(df)
        self.printTable(pt, "Finding optimal Theta using smap")

    def printNonLinearTestInfo(self):
        if self.nonlin_info is None:
            return
        
        pt = PrettyTable()

        resBase, resOpt, ksTest = self.nonlin_info

        thetaOpt = round(float(resOpt["theta"]), 3)
        rhoBase, rhoOpt = round(float(resBase["rho"]), 3), round(float(resOpt["rho"]), 3)
        ksStat, ksPVal = round(ksTest.statistic, 5), round(ksTest.pvalue, 5)

        pt.field_names = ["rho (theta=0)", f"rho (theta={thetaOpt})", "ks-stat", "p-value"]

        pt.add_row((rhoBase, rhoOpt, ksStat, ksPVal))

        self.printTable(pt, "Checking nonlinearity using Kolmogorov-Smirnov test")

    def printLagInfo(self):
        if self.lag_info is None:
            return
        
        pt = PrettyTable()
        pt.field_names = ["lag", "rho"]

        data = list(self.lag_info.items())
        data.sort(key=lambda x: -x[1])

        row_num = 0
        for _, tup in enumerate(data):
            lag, rho = tup
            lag = ("" if lag < 0 else "+") + str(lag)
            rho = round(rho, 5)
            pt.add_row((lag, rho))
            row_num += 1
            if row_num >= self.ROW_LIMIT:
                break

        self.printTable(pt, "Finding optimal lag using smap")

    def printConvergenceInfo(self):
        if self.conv_info is None:
            return
        
        pt = PrettyTable()
        type, dist, sample = self.conv_info

        if type == "quantile":
            # pt = PrettyTable()
            # pt.field_names = ['E', 'library', 'theta', 'rho', 'mae']
            # sorted_info = dist.sort_values(by=["rho"])
            # print(sorted_info)
            # row_num = 0
            # for _, row in sorted_info.iterrows():
            #     E, library, theta, rho, mae = row
            #     pt.add_row((int(E), int(library), round(theta, 5), round(rho, 5), round(mae, 5)))
            #     row_num += 1
            #     if row_num >= self.ROW_LIMIT:
            #         break

            # self.printTable(pt, "Testing convergence: distrbution at a small library size")

            finalRho = float(sample["rho"])
            rhoQuantile = np.count_nonzero(dist < finalRho) / dist.size

            pt = PrettyTable()
            pt.field_names = ["E", "library", "theta", "rho", "mae", "quantile"]

            row_num = 0
            for _, row in sample.iterrows():
                E, library, theta, rho, mae = row
                pt.add_row((int(E), int(library), round(theta, 5), round(rho, 5), round(mae, 5), round(rhoQuantile, 5)))
                row_num += 1
                if row_num >= self.ROW_LIMIT:
                    break

            self.printTable(pt, "Testing convergence: sampled at maximum library size")
