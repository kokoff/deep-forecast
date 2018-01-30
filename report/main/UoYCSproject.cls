% Jeremy Jacob                                           UoYCSproject

% A class file to typeset BEng, Bsc, MEng, MMath and taught MSc
% project reports in the Department of Computer Science at the
% University of York.

% History
%
% Version 1
%
% 2004Apr27  Initial version
% 2004May10  Generation of list of tables and figures removed
% 2004Dec08  Minor fixes to error messages
% 2005Feb02  Package pdfcprot replaced by microtype 
% 2005Apr12  dedication and acknowledgements now come after abstract
% 2005Jul27  added option `sc' to `mathpazo' package & `textcomp' package
% 2005Aug31  added packages for Helvetica and Courier fonts
%
% Version 2 created, not backward compatible with Version 1
%
% 2006Mar23  moved front matter generation from \AtBeginDocument to
%            separately called \maketitle
% 2006Mar23  added options for choice of citation style
% 2006May12  improved communication with type area calculation
% 2006Jul03  added hypcap package
% 2006Dec10  added fixltx2e package
% 2007May31  fixed autoref names
% 2007Nov01  fixed Masters title names
% 2007Nov12  Added old-style text figures.
% 2008Oct03  used hyperref option `pdfusetitle'
% 2008Oct27  removed page numbers from `Part' pages
% 2009Jun26  changed to IEEE trans style; removed options.
% 2010Nov02  replaced obsolete KOMA-script options
% 2014Oct22  added ACS declaration
% 
% Version 3, backwards compatible
%
% 2015Apr24 Converted from BibTeX to BibLaTeX
% 2016Dec05 Brought up to date with latest Koma Script

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{UoYCSproject}[2016/12/05 Jeremy Jacob]

% Koma-script report style
\PassOptionsToPackage{twoside}{typearea}
\LoadClass[fontsize=11pt,a4paper,twoside,abstracton,numbers=noenddot,BCOR13mm]{scrreprt}
\RequirePackage{scrhack}

\RequirePackage[T1]{fontenc} % prevents < in text mode turning into
\RequirePackage{textcomp}    % ?', etc
\RequirePackage[osf]{mathpazo} % Palatino font
\RequirePackage{courier} % nicer typewriter-style fonts
\RequirePackage[scaled=.9]{helvet} % nice san serif fonts
\RequirePackage{microtype} % micro-typographical extensions for
                           % pdf(e)(la)tex
\RequirePackage[british]{babel} % British hyphenation patterns, etc.
\RequirePackage{ifthen}

\pagestyle{headings}
\renewcommand{\partpagestyle}{empty}

% Reference control
\RequirePackage{csquotes}
\RequirePackage[style=ieee,backend=biber]{biblatex}


\typearea[current]{calc} % force page layout recalculation after
                         % changing all the fonts


\RequirePackage[pdfusetitle]{hyperref}% automatic hyperlinks, etc
\RequirePackage[all]{hypcap}% improve link placement in floats
\hypersetup{%
  pdfstartview={Fit}%
}
\urlstyle{sf}
\renewcommand*{\Itemautorefname}{Item}
\renewcommand*{\chapterautorefname}{Chapter}
\renewcommand*{\sectionautorefname}{Section}
\renewcommand*{\subsectionautorefname}{Subsection}
\renewcommand*{\subsubsectionautorefname}{Subsubsection}
\renewcommand*{\paragraphautorefname}{Paragraph}
\renewcommand*{\Hfootnoteautorefname}{Footnote}
%\renewcommand*{\pageautorefname}{Page}


\AtBeginDocument{%
  \nonfrenchspacing%
}
