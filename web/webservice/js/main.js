(function($){
    String.prototype.format = String.prototype.f = function() {
        var s = this,
            i = arguments.length;

        while (i--) {
            s = s.replace(new RegExp('\\{' + i + '\\}', 'gm'), arguments[i]);
        }
        return s;
    };

    var pmc_url = "http://www.ebi.ac.uk/europepmc/webservices/rest/search?query={0}%20open_access:y%20HAS_FT:y&format=json&cursorMark={1}&pageSize={2}";
    var nextCursorMark = "*";
    var offset = 0;
    var pageSize = 25;
    var total = 0;
    function setupTab(){
        $("li").click(function(e) {
            e.preventDefault();
            $("li").removeClass("selected");
            $("li").each(function(){
                $('#' + $(this).attr('linkdiv')).hide();
            });
            $(this).addClass("selected");
            var div = $(this).attr('linkdiv');
            if (div){
                $('#' + div).show();
            }
        });

        var div = $('.selected').attr('linkdiv');
        if (div){
            $('#' + div).show();
        }
    }

    function query(url, cb){
        $.ajax({
            dataType: "json",
            url: url,
            data: null,
            success: cb
        });
    }

    function searchPMC(){
        if ($.trim($('#termSearchInput').val()) == ""){
            swal('keyword please..');
        }else{
            offset = 0;
            nextCursorMark = '*';
            $('#paperList').html('');
            swal('searching PMC...');
            swal.showLoading();
            query(pmc_url.format($('#termSearchInput').val(), nextCursorMark, pageSize), function (d) {
                renderQueryResult(d);
            })
        }
    }

    function renderQueryResult(d){
        console.log(d);
        if (d && d['hitCount'] < 0){
            swal('no papers found!');
        }else {
            swal.close();
            total = parseInt(d['hitCount']);
            nextCursorMark = d['nextCursorMark'];

            var h = '<div class="pageCtrl"> ' + (Math.min(total, offset + 1)) + ' to ' + (Math.min(total, offset + pageSize)) + ' of total ' + d['hitCount'] + ' results <span class="btnNext">next</span></div>';
            var s = h;
            s += '<table id="papers"><tr><th width="140">PMCID</th><th>Paper</th></tr>';
            for (var i=0;i<d['resultList']['result'].length;i++){
                var r = d['resultList']['result'][i];
                s += '<tr>' +
                    '<td><label for="' + r['pmcid'] + '">' + r['pmcid'] + '</label> <input type="checkbox" id="' + r['pmcid'] + '" pmcid="' + r['pmcid'] + '" class="pmcIdCheck"/></td><td>' +
                    ('title'  in r ? r['title'] : "") + ' <br/>' + ('authorString'  in r ? r['authorString'] : "") +
                    ' <i>' + ('journalTitle'  in r ? r['journalTitle'] : "") + '</i> '
                    + ' (' + ('pubYear'  in r ? r['pubYear'] : "") + ')'
                    + '</td></tr>';
            }
            s += '</table>';
            s += h;
            $('#paperList').html(s);

            $('.pmcIdCheck').click(function () {
                var selected = [];
                $('input:checked').each(function() {
                    selected.push($(this).attr('pmcid'));
                });
                if (selected.length > 0 && selected.length <= 10){
                    $('.btnSubmitChecked').show();
                }else{
                    if (selected.length > 10)
                        swal('only 10 papers in one go, thanks!');
                    $('.btnSubmitChecked').hide();
                }
            });

            $('.btnNext').click(function(){
                $('#paperList').html('');
                swal('searching PMC...');
                swal.showLoading();
                query(pmc_url.format($('#termSearchInput').val(), nextCursorMark, pageSize), function (d) {
                    offset += pageSize;
                    renderQueryResult(d);
                });
            });
        };
    }

    function validate_pmcids(pmcids){
        if (pmcids.length < 200 && pmcids.match(/^PMC\d{3,10}(,PMC\d{3,10}){0,9}$/ig)){
            var arrPMCs = pmcids.split(",");
            if (arrPMCs.length <= 10){
                return true;
            }
        }
        return false;
    }

    function submitPMCIDs(pmids){
        var user = '';
        if (validate_pmcids(pmids)) {
            swal({
                title: 'Email(optional)?',
                text: 'Email for notification',
                input: 'email',
                inputValue: Cookies.get('userEmail') ? Cookies.get('userEmail') : '',
                showCancelButton: true,
                cancelButtonText: 'go without email'
            }).then(function (email) {
                user = email;
                doSubmitJob(user, pmids);
                Cookies.set('userEmail', email, { expires: 365 });
            }, function (dismiss) {
                // dismiss can be 'cancel', 'overlay',
                // 'close', and 'timer'
                if (dismiss === 'cancel') {
                    doSubmitJob(user, pmids);
                }
            })


        }else{
            swal('Error', 'invalid PMC IDs [' + pmids + ']', 'error');
        }
    }

    function doSubmitJob(user, pmids){
        swal('submitting PMC IDs...');
        swal.showLoading();
        try {
            qbb.inf.createHTJob(user, pmids, function (s) {
                console.log(s);
                if (s.match(/[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}/ig)) {
                    swal('Job created.', 'Job ID: ' + s, 'success').then(function () {
                        window.location = "ht.html?" + s;
                    });
                } else {
                    swal('Job creation failed. Please contact NAPEasy!');
                }
            }, function (jqXHR, textStatus, errorThrown) {
                swal('job creation failed', 'up to 5 jobs in 10 mins per user, try later.', 'error');

            });
        }catch (err){
            console.log(err);
            swal('job creation failed, try in 10 mins');
        }
    }

	$(document).ready(function(){
        setupTab();

        $('pre code').each(function(i, block) {
            hljs.highlightBlock(block);
        });

        $('.btnPMCSubmit').click(function () {
            submitPMCIDs($('#pmcidText').val());
        });

        $('.btnSubmitChecked').click(function () {
            var selected = [];
            $('input:checked').each(function() {
                selected.push($(this).attr('pmcid'));
            });
            submitPMCIDs(selected.join(','));
        });

        $('.btnSearch').click(function () {
            searchPMC();
        });

        $('#termSearchInput').keypress(function (e) {
            if (e.which == 13) {
                $('.btnSearch').click();
            }
        });

        $('#helpSwitch').click(function () {
            $(this).hide();
            $('#help').slideDown();
        });

        $('.btnHideHelp').click(function(){
            $('#help').slideUp("", function(){
                $('#helpSwitch').show();
            });
        })
        
        $('.pmcidSample').click(function () {
            $('#pmcidText').val($(this).html());
        })

        $('.keyworkSample').click(function () {
            $('#termSearchInput').val($(this).html());
        })
	})

})(this.jQuery)