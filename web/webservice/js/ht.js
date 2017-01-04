(function($){
    var _jobID = null;
    var _sentencens = null;

    String.prototype.format = String.prototype.f = function() {
        var s = this,
            i = arguments.length;

        while (i--) {
            s = s.replace(new RegExp('\\{' + i + '\\}', 'gm'), arguments[i]);
        }
        return s;
    };

    function getJOBStatus(jo){
        var curStatus = jo['currentStatus'];
        if (curStatus == null){
            return '[{0}] job created, waiting in the queue.'.format(jo['createdDate']);
        }else if (curStatus == '100'){
            return '[{0}] job is being processed.'.format(jo['updatedDate'])
        }else if (curStatus == '101'){
            return '[{0}] job is being processed.'.format(jo['updatedDate'])
        }else if (curStatus == '102'){
            return '[{0}] job is being processed. Scoring sentences...'.format(jo['updatedDate'])
        }else if (curStatus == '103'){
            return '[{0}] job is being processed. Doing semantic calculation...'.format(jo['updatedDate'])
        }else if (curStatus == '104'){
            return '[{0}] job is being processed. Highlighting sentences...'.format(jo['updatedDate'])
        }else if (curStatus == '501'){
            return '[{0}] job process encountered errors. Filesystem error.'.format(jo['updatedDate'])
        }else if (curStatus == '502'){
            return '[{0}] job process encountered errors. Highlighting failed.'.format(jo['updatedDate'])
        }else if (curStatus == '200'){
            return '[{0}] job finished successfully'.format(jo['updatedDate'])
        }
    }

    function getUrlVars()
    {
        var vars = [], hash;
        var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
        for(var i = 0; i < hashes.length; i++)
        {
            hash = hashes[i].split('=');
            vars.push(hash[0]);
            vars[hash[0]] = hash[1];
        }
        return vars;
    }

    var _htResults = null;
    var _curDocIdx = 0;
    var _curPaperId = null;
    var _showMarked = true;

    function loadFullText(pmid){
        qbb.inf.getPaperFullText(pmid, function(s){
            if (s) {
                renderFullText($.parseJSON(s));
            }else
                swal('full text not found on server');
        })
    }

    function loadCurrentPaper(){
        _curPaperId = _htResults[_curDocIdx]['pmcid'];
        loadFullText(_curPaperId);
        $('input[type="radio"]').prop('checked', false);
    }

    function renderPageControl(){
        $('.pageInfo').html( (_curDocIdx + 1) + '/' + _htResults.length + ' papers');
    }

    function renderPrediction(paper){
        $('.clsHT').removeClass('clsHT');
        for (var i=0;i<paper.highlights.length;i++){
            if (paper.highlights[i].length > 2 && paper.highlights[i][2].length > 0)
                $('.clsType.s' + paper.highlights[i][0]).html(paper.highlights[i][2])
                // $( '<span class="triangle-border pop">' + paper.highlights[i][2] + '</span>' ).insertBefore($('div[sid="' + paper.highlights[i][0] + '"]'));
            $('div[sid="' + paper.highlights[i][0] + '"]').addClass('clsHT');
            $('div[sid="' + paper.highlights[i][0] + '"] .clsSentText').addClass('clsHT');
        }

        if (!$('.btnType').prop('checked'))
            $('.clsType').hide();
        if ($('.btnHideNormal').prop('checked')){
            $('.normal').hide();
            $('.clsHT').show();
        }
        $('#visPanel').show();
    }

    function loadTypedHighlights(pp){
        swal('complementary highlighting...');
        qbb.inf.getPaperSumm(pp.pmcid, function (s) {
            swal.close();
            var so = $.parseJSON(s);
            console.log(s);
            var paper = {'pmcid': pp.pmcid, 'highlights': []}
            for (var t in so){
                for (var i=0;i<so[t].length;i++){
                    paper.highlights.push([
                        so[t][i][1]['sid'],
                        so[t][i][1]['total'],
                        t
                    ]);
                }
            }
            renderPrediction(paper);
        });
    }

    function renderFullText(sents){
        _sentencens = sents;
        renderPageControl();
        var s = '';
        if (sents.length > 0)
            $('.pageTitle').html(sents[0].text);
        for (var i=1;i<sents.length;i++){
            var bMarked = 'marked' in sents[i];
            var cls = 'normal';
            var blur = '';
            if ($('.btnBlur').prop('checked')){
                blur = ' blur';
            }
            if (_showMarked){
                cls = bMarked?'marked ' + cls:cls;
            }
            s +=  '<div sid="' + sents[i].sid + '" class="' + cls + '">'
                    + '<div class="clsSid">' + (parseInt(sents[i].sid) - 1) + '.</div>'
                    + '<div class="clsSentText ' + blur + '">'
                    + sents[i].text + '</div>'
                    + '<div class="clsType s'+ sents[i].sid + '">&nbsp;</div>'
                    + '</div>';
        }
        $('.clsFullText').html(s);
        renderCurrentHighlights();
    }

    function renderCurrentHighlights(){
        $('.clsFullText').scrollTop(0);
        var paper = _htResults[_curDocIdx];
        $('.pmcidCtn').html("[{0}]".format(paper.pmcid));
        if (paper.highlights.length<=2 && !$('.btnThreshold').prop('checked')){
            loadTypedHighlights(paper);
        }else {
            renderPrediction(paper);
        }
    }

    function checkJobStatus(){
        if (qbb.inf.isValidJobId(_jobID)) {
            swal('checking...');
            qbb.inf.getJobDetail(_jobID, function(s){
                if (s){
                    console.log(s);
                    swal.close();
                    var jo = $.parseJSON(s);
                    $('#jobId').html('Job ID: ' + _jobID);
                    $('#status').html(getJOBStatus(jo));
                    var errorPapers = [];
                    if (jo['currentStatus'] == "200"){
                        var pm2ht = jo['hts'];
                        _htResults = [];
                        for (var pm in pm2ht){
                            var hto = $.parseJSON(pm2ht[pm]);
                            if (hto) {
                                hto['pmcid'] = pm;
                                _htResults.push(hto);
                            }else
                                errorPapers.push(pm);
                        }
                        if (_htResults.length > 0) {
                            loadCurrentPaper();

                            $('.clsPageLink').click(function () {
                                if ($(this).html() == 'prev') {
                                    if (_curDocIdx <= 0) {
                                        _curDocIdx = _htResults.length > 0 ? _htResults.length - 1 : 0;
                                    } else
                                        _curDocIdx--;
                                } else {
                                    if (_curDocIdx >= _htResults.length - 1) {
                                        _curDocIdx = 0;
                                    } else
                                        _curDocIdx++;
                                }
                                loadCurrentPaper();
                            });
                        }
                    }
                    if (errorPapers.length > 0)
                        $('#status').html(getJOBStatus(jo) + ' | full texts of [' + errorPapers.join(',') + '] not available.');
                }else{
                    swal('job not exists');
                }

            });
        }else{
            swal('invalid job id');
        }
    }

    function ht2xml(ht){
        $jobElem = $('<job></job>').hide().attr('jobid', _jobID);
        for (var i=0;i<ht.length;i++){
            var p = ht[i];
            var $pElem = $('<paper></paper>')
                .attr('pmcid', p.pmcid).attr('total_sentences', parseInt(p.max_sid) + 1);
            $jobElem.append($pElem);
            var sents = p.highlights.sort(function(a, b){
               return parseInt(a[0]) - parseInt(b[0]);
            });
            for (var j=0;j<sents.length;j++){
                var hls = sents[j];
                $pElem.append(
                    $('<highlight></highlight>')
                        .attr('sid', hls[0])
                        .attr('type', hls[2])
                        .attr('score', hls[1])
                        .html(_sentencens[parseInt(hls[0] - 1)].text)
                );
            }
        }
        return '<?xml version="1.0" encoding="UTF-8"?>' + $('<pp></pp>').append($jobElem).html();
    }

    function dumpJSON(ht){
        var job = {'jobid': _jobID, 'papers': []};
        for (var i=0;i<ht.length;i++){
            var p = ht[i];
            var pp = {"highlights": []};
            job.papers.push(pp);
            pp.pmcid = p.pmcid;
            pp.total_sentences = parseInt(p.max_sid) + 1;
            p.highlights.sort(function(a, b){
                return parseInt(a[0]) - parseInt(b[0]);
            });
            for (var j=0;j<p.highlights.length;j++){
                var hls = p.highlights[j];
                pp.highlights.push({
                    "sid": hls[0],
                    "text": _sentencens[parseInt(hls[0])-1].text,
                    "type": hls[2],
                    "score": hls[1]
                });
            }
        }
        return $.toJSON(job);
    }

    function dumpText(ht){
        var arr = ['jobid - ' + _jobID];
        for (var i=0;i<ht.length;i++){
            var p = ht[i];
            var pp = {"highlights": []};
            arr.push(p.pmcid + ' (#sentences: ' + (parseInt(p.max_sid) + 1) + ')');
            p.highlights.sort(function(a, b){
                return parseInt(a[0]) - parseInt(b[0]);
            });
            for (var j=0;j<p.highlights.length;j++){
                var hls = p.highlights[j];
                arr.push('[sid: ' + hls[0] + ', type:' + hls[2] + ', score: ' + hls[1] + '] ' + _sentencens[parseInt(hls[0])-1].text);
            }
        }
        return arr.join('\n');
    }

	$(document).ready(function(){
	    _jobID = window.location.href.slice(window.location.href.indexOf('?') + 1);
        $('.btnBlur').prop('checked', Cookies.get('blurCheck')=="true");
        $('.btnType').prop('checked', Cookies.get('typeCheck')=="true");
        $('.btnHideNormal').prop('checked', Cookies.get('hideNormalCheck')=="true");
        $('.btnThreshold').prop('checked', Cookies.get('threshold')=="true");
        checkJobStatus();

        // loadHighlighted();

        $('.btnBlur').click(function(){
            if ($(this).prop('checked'))
                $('.clsSentText').addClass('blur');
            else
                $('.clsSentText').removeClass('blur');
            $('.clsHT').removeClass('blur');
            Cookies.set('blurCheck', $(this).prop('checked'), { expires: 365 });
        });

        $('.btnType').click(function(){
            if ($(this).prop('checked'))
                $('.clsType').show();
            else
                $('.clsType').hide();
            Cookies.set('typeCheck', $(this).prop('checked'), { expires: 365 });
        });

        $('.btnHideNormal').click(function(){
            if ($(this).prop('checked'))
                $('.normal').hide();
            else
                $('.normal').show();
            $('.clsHT').show();

            Cookies.set('hideNormalCheck', $(this).prop('checked'), { expires: 365 });
        });

        $('.btnThreshold').click(function(){
            renderCurrentHighlights();

            Cookies.set('threshold', $(this).prop('checked'), { expires: 365 });
        });

        $('.download').click(function(){
            if (!_sentencens)
                swal('data not ready.')
            var url = 'http://napeasy.org/napeasy_api/download/';
            $('#_dform').attr('action', url);
            $('#dform').val($(this).html());
            var data = null;
            if ($(this).html() == "JSON"){
                data = dumpJSON(_htResults);
            }else if ($(this).html() == "XML"){
                data = ht2xml(_htResults);
            }else{
                data = dumpText(_htResults);
            }
            $('#ddata').val(data);
            $('#_dform').submit();
        })
	})

})(this.jQuery)