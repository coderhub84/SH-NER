# SH-NER

\begin{table}[ht]
\small  % or \scriptsize to reduce font size further
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{8.0pt}
\begin{tabular}{lrrrr}
\hline
\textbf{Entity Type} & \textbf{Train} & \textbf{\%} & \textbf{Test} & \textbf{\%} \\
\hline
Software-Entity   & 2,245 & 42.46 & 281 & 42.38 \\
Hardware-device   & 1,478 & 27.95 & 174  & 26.24 \\
Device-Count      & 881   & 16.66 & 118  & 17.79 \\
Device-Memory     & 507   & 9.6 & 52  & 8.00 \\
Cloud-Platform    & 176   & 3.32  & 38  & 5.73  \\
\hline
\textbf{Total}    & 5,287 & 100  & 663 & 100  \\
\hline
\end{tabular}
\caption{Distribution of entity types across training and test sets in the SH-NER dataset, showing both counts and relative percentages.}
\label{tab:entity-dist}
\end{table}
