class DefaultConfig:
    #columns_categorical = ['_CH_BirthOrder_0d']
    # age 18 if run cleaner first
    #columns_categorical = [
    #    'c804_32wg', 'c800_32wg',
    #    'c801_32wg', 'kz021_0m',
    #    'b122r_18wg', 'b123_18wg',
    #    'c101_32wg', 'd536a_12wg',
    #    'd586a_12wg', 'd171a_12wg',
    #    'e327_8w', 'f021a_8m',
    #    'f063a_8m', 'b107_18wg',
    #    'c093_32wg', 'f020a_8m',
    #    'f229a_8m', 'kb066_6m',
    #    'kd057_18m', 'kd510b_18m',
    #    'e411r_8w', 'e414r_8w',
    #    'f251a_8m', 'f252a_8m',
    #    'f260a_8m', 'f635r_8m',
    #    'a522_8wg', 'neverm8wg_8wg',
    #    'married8wg_8wg', 'widowed8wg_8wg',
    #    'divorced8wg_8wg', 'separated8wg_8wg',
    #    'c760_32wg', 'f220a_8m',
    #    'f228a_8m', 'f235a_8m',
    #    'f237a_8m', 'f245a_8m',
    #    'neverm8m_8m', 'married8m_8m',
    #    'widowed8m_8m', 'divorced8m_8m',
    #    'separated8m_8m', 'f500_8m',
    #    'f501_8m']
    # should work for everything but just in case, run get_config_info.py
    columns_categorical = ['c804_32wg', 'c800_32wg', 'c801_32wg', 'kz021_0m', 'kr800_91m', 'kr801_91m', 'kr802_91m', 'kr803a_91m', 'kr810_91m', 'kr811_91m', 'kr812_91m', 'kr813a_91m', 'kr815_91m', 'kr820_91m', 'kr821_91m', 'kr822_91m', 'kr823_91m', 'kr824_91m', 'kr825_91m', 'kr826_91m', 'kr827a_91m', 'kr830_91m', 'kr831_91m', 'kr832a_91m', 'ku421_108m', 'ku423r_108m', 'DEL_P1490_1m', 'b122r_18wg', 'b123_18wg', 'c101_32wg', 'd536a_12wg', 'd586a_12wg', 'd171a_12wg', 'e327_8w', 'f021a_8m', 'f063a_8m', 'f518a_8m', 'g021a_21m', 'g049a_21m', 'g120a_21m', 'g604a_21m', 'h013a_33m', 'h039a_33m', 'h090a_33m', 'h489a_33m', 'j012a_47m', 'j044a_47m', 'j100a_47m', 'j607a_47m', 'k1011r_61m', 'l3011r_73m', 'l6023r_73m', 'n1061r_97m', 'n2013_97m', 'n2033_97m', 'p1011r_110m', 'p3023r_110m', 'pd021a_8m', 'pd063a_8m', 'pe021a_21m', 'pe064a_21m', 'pe130a_21m', 'pf1011r_33m', 'pg1011r_47m', 'ph1011r_61m', 'pl1061r_97m', 'pm1011r_110m', 'pm3023r_110m', 'pd020a_8m', 'pe020a_21m', 'pf1010r_33m', 'pg1010r_47m', 'pm1010r_110m', 'pm3024r_110m', 'b107_18wg', 'c093_32wg', 'f020a_8m', 'f519a_8m', 'g020a_21m', 'g605a_21m', 'h012a_33m', 'h490a_33m', 'k1010r_61m', 'l3010r_73m', 'p1010r_110m', 'p3024r_110m', 'f229a_8m', 'h219a_33m', 'kb066_6m', 'kb069_6m', 'kd057_18m', 'kd510b_18m', 'kf075_30m', 'kj470a_42m', 'kl045_57m', 'kn1050_69m', 'kq040_81m', 'kq371a_81m', 'ks1070_103m', 'reprodpb_97m', 'sa033a_90m', 'sa034a_90m', 'sa035a_90m', 'sa036a_90m', 'sa037a_90m', 'sa038a_90m', 'sa039a_90m', 'sa040a_90m', 'sa042b_90m', 'sa043b_90m', 'sa045a_90m', 'sa045b_90m', 'sa047_90m', 'sa048_90m', 'sa049_90m', 'sa050_90m', 'sa054_90m', 'b581r_18wg', 'b584r_18wg', 'e411r_8w', 'e414r_8w', 'g311a_21m', 'g314a_21m', 'h221a_33m', 'h224a_33m', 'j311a_47m', 'j314a_47m', 'k4011r_61m', 'k4014r_61m', 'l4011r_73m', 'p2011r_110m', 'p2014r_110m', 'f251a_8m', 'f252a_8m', 'f260a_8m', 'f566_8m', 'f635r_8m', 'g331a_21m', 'g332a_21m', 'g340a_21m', 'g650_21m', 'g652_21m', 'g764r_21m', 'h241a_33m', 'h242a_33m', 'h250a_33m', 'h530_33m', 'h534_33m', 'j331a_47m', 'j332a_47m', 'j340a_47m', 'j350_47m', 'j351_47m', 'j632_47m', 'j633_47m', 'k6000r_61m', 'k6264_61m', 'l6080_73m', 'l6083_73m', 'n4140_97m', 'p2031r_110m', 'p2032r_110m', 'p2042r_110m', 'p3080_110m', 'p3083_110m', 'q5000r_122m', 'kr422b_91m', 'a522_8wg', 'neverm8wg_8wg', 'married8wg_8wg', 'widowed8wg_8wg', 'divorced8wg_8wg', 'separated8wg_8wg', 'b570r_18wg', 'b578r_18wg', 'b585r_18wg', 'b587r_18wg', 'b595r_18wg', 'c760_32wg', 'f220a_8m', 'f228a_8m', 'f235a_8m', 'f237a_8m', 'f245a_8m', 'neverm8m_8m', 'married8m_8m', 'widowed8m_8m', 'divorced8m_8m', 'separated8m_8m', 'f500_8m', 'f501_8m', 'g300a_21m', 'g308a_21m', 'g315a_21m', 'g317a_21m', 'g325a_21m', 'neverm21m_21m', 'married21m_21m', 'widowed21m_21m', 'divorced21m_21m', 'separated21m_21m', 'g520r_21m', 'g590r_21m', 'g591_21m', 'h210a_33m', 'h218a_33m', 'h225a_33m', 'h227a_33m', 'h235a_33m', 'neverm33m_33m', 'married33m_33m', 'widowed33m_33m', 'divorced33m_33m', 'separated33m_33m', 'h480r_33m', 'h481_33m', 'j300a_47m', 'j308a_47m', 'j315a_47m', 'j317a_47m', 'j325a_47m', 'j600_47m', 'j601_47m', 'l4000r_73m', 'separated73m_73m', 'l6000r_73m', 'l6001_73m', 'neverm85m_85m', 'married85m_85m', 'widowed85m_85m', 'divorced85m_85m', 'separated85m_85m', 'm3100_85m', 'm3202r_85m', 'n4160_97m', 'married97m_97m', 'widowed97m_97m', 'divorced97m_97m', 'separated97m_97m', 'p2000r_110m', 'p2008r_110m', 'p2015r_110m', 'p2017r_110m', 'p2025r_110m', 'p3000r_110m', 'p3001_110m', 'p3003r_110m', 'neverm122m_122m', 'married122m_122m', 'widowed122m_122m', 'divorced122m_122m', 'separated122m_122m', 'Widowed01_122m', 'Divorced01_122m']